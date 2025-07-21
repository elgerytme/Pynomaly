"""
Strategic Quality Analytics Service
Advanced analytics engine with predictive capabilities for quality management.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
import pickle
from pathlib import Path

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from ...domain.entities.executive_scorecard import (
    QualityMaturityStage, TrendDirection, QualityKPI, BusinessUnit,
    IndustryBenchmark, FinancialImpact, QualityROIAnalysis
)
from ...domain.entities.data_quality_assessment import QualityAssessment


@dataclass
class PredictionConfig:
    """Configuration for predictive analytics."""
    
    # Model selection
    prediction_algorithm: str = "gradient_boosting"  # linear, random_forest, gradient_boosting
    feature_selection_threshold: float = 0.01
    
    # Time series parameters
    lookback_window_days: int = 90
    prediction_horizon_days: int = 30
    seasonal_adjustment: bool = True
    
    # Training parameters
    train_test_split_ratio: float = 0.8
    cross_validation_folds: int = 5
    min_historical_points: int = 10
    
    # Performance thresholds
    min_model_r2_score: float = 0.6
    max_prediction_uncertainty: float = 0.2
    
    # Update frequency
    model_retrain_interval_days: int = 7
    prediction_update_interval_hours: int = 6


@dataclass
class QualityPrediction:
    """Quality prediction result."""
    
    metric_name: str
    prediction_timestamp: datetime
    prediction_horizon_days: int
    
    # Predicted values
    predicted_value: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    prediction_confidence: float
    
    # Model performance
    model_accuracy: float
    historical_variance: float
    trend_strength: float
    
    # Risk assessment
    risk_level: str  # low, medium, high, critical
    risk_factors: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    
    # Supporting data
    feature_importance: Dict[str, float] = field(default_factory=dict)
    similar_historical_patterns: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class InvestmentOptimization:
    """Investment optimization recommendation."""
    
    optimization_id: str
    investment_scenario: str
    recommended_budget: float
    expected_roi: float
    confidence_score: float
    
    # Investment allocation
    area_allocations: Dict[str, float] = field(default_factory=dict)  # technology, people, process, tools
    priority_initiatives: List[str] = field(default_factory=list)
    
    # Expected outcomes
    quality_improvement_projection: float
    risk_reduction_projection: float
    cost_savings_projection: float
    
    # Timeline
    implementation_timeline_months: int = 12
    payback_period_months: Optional[int] = None
    
    # Justification
    rationale: str = ""
    supporting_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompetitiveAnalysis:
    """Competitive quality analysis."""
    
    analysis_id: str
    comparison_date: datetime
    organization_position: str  # leader, challenger, follower, niche
    
    # Competitive metrics
    market_position_score: float
    quality_maturity_gap: float
    investment_gap_percentage: float
    
    # Benchmarking
    peer_comparison: Dict[str, float] = field(default_factory=dict)
    industry_leaders: List[str] = field(default_factory=list)
    best_practices_gaps: List[str] = field(default_factory=list)
    
    # Strategic insights
    competitive_advantages: List[str] = field(default_factory=list)
    improvement_opportunities: List[str] = field(default_factory=list)
    threat_assessment: List[str] = field(default_factory=list)


class StrategicQualityAnalyticsService:
    """
    Advanced analytics service for strategic quality management with predictive capabilities.
    """
    
    def __init__(self, config: PredictionConfig = None):
        self.config = config or PredictionConfig()
        self.logger = logging.getLogger(__name__)
        
        # Model storage
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Historical data storage
        self.historical_data: Dict[str, pd.DataFrame] = {}
        
        # Cache for predictions
        self.prediction_cache: Dict[str, QualityPrediction] = {}
        self.cache_expiry: Dict[str, datetime] = {}
        
        # Feature engineering
        self.feature_generators = self._initialize_feature_generators()
        
        if not SKLEARN_AVAILABLE:
            self.logger.warning("Scikit-learn not available. Predictive features will use statistical methods.")
    
    def _initialize_feature_generators(self) -> Dict[str, callable]:
        """Initialize feature generation functions."""
        return {
            "temporal_features": self._generate_temporal_features,
            "trend_features": self._generate_trend_features,
            "seasonal_features": self._generate_seasonal_features,
            "lag_features": self._generate_lag_features,
            "statistical_features": self._generate_statistical_features,
            "business_context_features": self._generate_business_context_features
        }
    
    async def predict_quality_metrics(
        self,
        metric_names: List[str],
        historical_assessments: List[QualityAssessment],
        business_context: Dict[str, Any] = None,
        prediction_horizon_days: int = None
    ) -> Dict[str, QualityPrediction]:
        """
        Predict future quality metrics using historical data and advanced analytics.
        
        Args:
            metric_names: List of quality metrics to predict
            historical_assessments: Historical quality assessment data
            business_context: Additional business context for predictions
            prediction_horizon_days: How far ahead to predict (overrides config)
            
        Returns:
            Dictionary of metric predictions
        """
        try:
            self.logger.info(f"Starting quality prediction for {len(metric_names)} metrics")
            
            horizon = prediction_horizon_days or self.config.prediction_horizon_days
            business_context = business_context or {}
            
            # Prepare historical data
            historical_df = self._prepare_historical_data(historical_assessments)
            
            predictions = {}
            
            for metric_name in metric_names:
                # Check cache first
                cache_key = f"{metric_name}_{horizon}"
                if self._is_prediction_cached(cache_key):
                    predictions[metric_name] = self.prediction_cache[cache_key]
                    continue
                
                # Generate prediction
                prediction = await self._predict_single_metric(
                    metric_name, historical_df, horizon, business_context
                )
                
                predictions[metric_name] = prediction
                
                # Cache result
                self.prediction_cache[cache_key] = prediction
                self.cache_expiry[cache_key] = datetime.now() + timedelta(
                    hours=self.config.prediction_update_interval_hours
                )
            
            self.logger.info(f"Completed quality predictions for {len(predictions)} metrics")
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error in quality prediction: {str(e)}")
            raise
    
    async def optimize_quality_investment(
        self,
        current_quality_scores: Dict[str, float],
        available_budget: float,
        improvement_targets: Dict[str, float],
        business_constraints: Dict[str, Any] = None
    ) -> InvestmentOptimization:
        """
        Optimize quality investment allocation using predictive analytics.
        
        Args:
            current_quality_scores: Current quality metric scores
            available_budget: Available investment budget
            improvement_targets: Target improvement levels
            business_constraints: Business constraints and priorities
            
        Returns:
            Investment optimization recommendation
        """
        try:
            self.logger.info(f"Optimizing quality investment for ${available_budget:,.0f} budget")
            
            business_constraints = business_constraints or {}
            
            # Calculate improvement gaps
            improvement_gaps = {
                metric: max(0, target - current_quality_scores.get(metric, 0))
                for metric, target in improvement_targets.items()
            }
            
            # Estimate investment requirements
            investment_requirements = await self._estimate_investment_requirements(
                improvement_gaps, business_constraints
            )
            
            # Optimize allocation
            optimal_allocation = await self._optimize_investment_allocation(
                available_budget, investment_requirements, business_constraints
            )
            
            # Calculate expected outcomes
            expected_outcomes = await self._calculate_investment_outcomes(
                optimal_allocation, improvement_gaps
            )
            
            # Generate recommendations
            optimization = InvestmentOptimization(
                optimization_id=f"opt_{int(datetime.now().timestamp())}",
                investment_scenario="optimized_allocation",
                recommended_budget=available_budget,
                expected_roi=expected_outcomes.get("roi", 0.0),
                confidence_score=expected_outcomes.get("confidence", 0.7),
                area_allocations=optimal_allocation["areas"],
                priority_initiatives=optimal_allocation["initiatives"],
                quality_improvement_projection=expected_outcomes.get("quality_improvement", 0.0),
                risk_reduction_projection=expected_outcomes.get("risk_reduction", 0.0),
                cost_savings_projection=expected_outcomes.get("cost_savings", 0.0),
                implementation_timeline_months=optimal_allocation.get("timeline_months", 12),
                payback_period_months=expected_outcomes.get("payback_months"),
                rationale=self._generate_investment_rationale(optimal_allocation, expected_outcomes),
                supporting_data=expected_outcomes
            )
            
            self.logger.info(f"Investment optimization completed with {optimization.expected_roi:.1f}% expected ROI")
            return optimization
            
        except Exception as e:
            self.logger.error(f"Error in investment optimization: {str(e)}")
            raise
    
    async def analyze_competitive_position(
        self,
        organization_metrics: Dict[str, float],
        industry_benchmarks: List[IndustryBenchmark],
        market_context: Dict[str, Any] = None
    ) -> CompetitiveAnalysis:
        """
        Analyze competitive position in data quality management.
        
        Args:
            organization_metrics: Organization's quality metrics
            industry_benchmarks: Industry benchmark data
            market_context: Additional market context
            
        Returns:
            Competitive analysis results
        """
        try:
            self.logger.info("Starting competitive quality analysis")
            
            market_context = market_context or {}
            
            # Calculate market position
            market_position = self._calculate_market_position(organization_metrics, industry_benchmarks)
            
            # Analyze maturity gap
            maturity_gap = self._analyze_maturity_gap(organization_metrics, industry_benchmarks)
            
            # Calculate investment gap
            investment_gap = self._calculate_investment_gap(
                organization_metrics, industry_benchmarks, market_context
            )
            
            # Identify competitive advantages and gaps
            advantages, gaps = self._identify_competitive_advantages_and_gaps(
                organization_metrics, industry_benchmarks
            )
            
            # Generate strategic insights
            threats = self._assess_competitive_threats(market_position, maturity_gap, market_context)
            opportunities = self._identify_improvement_opportunities(gaps, market_context)
            
            analysis = CompetitiveAnalysis(
                analysis_id=f"comp_{int(datetime.now().timestamp())}",
                comparison_date=datetime.now(),
                organization_position=market_position["position"],
                market_position_score=market_position["score"],
                quality_maturity_gap=maturity_gap,
                investment_gap_percentage=investment_gap,
                peer_comparison=market_position["peer_scores"],
                industry_leaders=market_position["leaders"],
                best_practices_gaps=gaps,
                competitive_advantages=advantages,
                improvement_opportunities=opportunities,
                threat_assessment=threats
            )
            
            self.logger.info(f"Competitive analysis completed - Position: {analysis.organization_position}")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in competitive analysis: {str(e)}")
            raise
    
    async def generate_strategic_recommendations(
        self,
        quality_predictions: Dict[str, QualityPrediction],
        investment_optimization: InvestmentOptimization,
        competitive_analysis: CompetitiveAnalysis,
        business_objectives: Dict[str, Any] = None
    ) -> List[str]:
        """
        Generate strategic recommendations based on comprehensive analytics.
        """
        try:
            recommendations = []
            business_objectives = business_objectives or {}
            
            # Quality trend recommendations
            for metric, prediction in quality_predictions.items():
                if prediction.risk_level in ["high", "critical"]:
                    recommendations.extend([
                        f"Immediate attention required for {metric} - predicted decline of {abs(prediction.predicted_value - 1.0)*100:.1f}%",
                        f"Implement {', '.join(prediction.recommended_actions[:2])} to mitigate {metric} risks"
                    ])
                elif prediction.trend_strength > 0.8:
                    recommendations.append(
                        f"Leverage positive trend in {metric} - consider expanding successful practices"
                    )
            
            # Investment recommendations
            if investment_optimization.expected_roi > 200:
                recommendations.append(
                    f"High-ROI investment opportunity identified: {investment_optimization.expected_roi:.0f}% expected return"
                )
            
            top_investment_areas = sorted(
                investment_optimization.area_allocations.items(),
                key=lambda x: x[1], reverse=True
            )[:2]
            
            recommendations.extend([
                f"Prioritize investment in {area}: ${amount:,.0f} allocation"
                for area, amount in top_investment_areas
            ])
            
            # Competitive recommendations
            if competitive_analysis.organization_position in ["follower", "niche"]:
                recommendations.extend([
                    "Accelerate quality maturity development to improve competitive position",
                    f"Address key gaps: {', '.join(competitive_analysis.best_practices_gaps[:3])}"
                ])
            
            if competitive_analysis.quality_maturity_gap > 0.3:
                recommendations.append(
                    "Significant maturity gap identified - consider strategic partnerships or acquisitions"
                )
            
            # Strategic focus recommendations
            if business_objectives.get("growth_focus"):
                recommendations.append(
                    "Align quality investments with growth objectives - focus on scalability and automation"
                )
            
            if business_objectives.get("cost_optimization"):
                recommendations.append(
                    "Emphasize cost-effective quality improvements with short payback periods"
                )
            
            # Risk mitigation recommendations
            high_risk_predictions = [
                name for name, pred in quality_predictions.items()
                if pred.risk_level in ["high", "critical"]
            ]
            
            if len(high_risk_predictions) > 2:
                recommendations.append(
                    "Multiple high-risk quality metrics identified - consider comprehensive quality overhaul"
                )
            
            return recommendations[:10]  # Return top 10 recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating strategic recommendations: {str(e)}")
            return ["Error generating recommendations - please review analytics manually"]
    
    # Private methods for prediction logic
    
    def _prepare_historical_data(self, assessments: List[QualityAssessment]) -> pd.DataFrame:
        """Prepare historical assessment data for analysis."""
        
        data = []
        for assessment in assessments:
            data.append({
                'timestamp': assessment.assessment_timestamp,
                'overall_score': assessment.overall_score,
                'completeness_score': assessment.completeness_score,
                'accuracy_score': assessment.accuracy_score,
                'consistency_score': assessment.consistency_score,
                'validity_score': assessment.validity_score,
                'uniqueness_score': assessment.uniqueness_score,
                'timeliness_score': assessment.timeliness_score
            })
        
        df = pd.DataFrame(data)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            df = df.set_index('timestamp')
        
        return df
    
    async def _predict_single_metric(
        self,
        metric_name: str,
        historical_df: pd.DataFrame,
        horizon_days: int,
        business_context: Dict[str, Any]
    ) -> QualityPrediction:
        """Predict a single quality metric."""
        
        if historical_df.empty or metric_name not in historical_df.columns:
            return self._create_default_prediction(metric_name, horizon_days)
        
        # Extract time series for the metric
        metric_series = historical_df[metric_name].dropna()
        
        if len(metric_series) < self.config.min_historical_points:
            return self._create_default_prediction(metric_name, horizon_days)
        
        # Generate features
        features_df = await self._generate_prediction_features(
            metric_series, business_context
        )
        
        if SKLEARN_AVAILABLE and len(features_df) >= self.config.min_historical_points:
            return await self._ml_prediction(metric_name, features_df, horizon_days)
        else:
            return await self._statistical_prediction(metric_name, metric_series, horizon_days)
    
    async def _generate_prediction_features(
        self,
        metric_series: pd.Series,
        business_context: Dict[str, Any]
    ) -> pd.DataFrame:
        """Generate features for prediction model."""
        
        df = pd.DataFrame({'value': metric_series})
        
        # Generate all feature types
        for feature_type, generator in self.feature_generators.items():
            try:
                features = generator(df, business_context)
                df = pd.concat([df, features], axis=1)
            except Exception as e:
                self.logger.warning(f"Failed to generate {feature_type}: {str(e)}")
        
        # Remove any columns with all NaN values
        df = df.dropna(axis=1, how='all')
        
        # Forward fill remaining NaN values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df
    
    def _generate_temporal_features(self, df: pd.DataFrame, context: Dict[str, Any]) -> pd.DataFrame:
        """Generate temporal features."""
        features = pd.DataFrame(index=df.index)
        
        features['day_of_week'] = df.index.dayofweek
        features['day_of_month'] = df.index.day
        features['month'] = df.index.month
        features['quarter'] = df.index.quarter
        features['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        features['is_month_end'] = (df.index.day >= 28).astype(int)
        features['is_quarter_end'] = df.index.month.isin([3, 6, 9, 12]).astype(int)
        
        return features
    
    def _generate_trend_features(self, df: pd.DataFrame, context: Dict[str, Any]) -> pd.DataFrame:
        """Generate trend features."""
        features = pd.DataFrame(index=df.index)
        
        # Moving averages
        features['ma_7'] = df['value'].rolling(window=7, min_periods=1).mean()
        features['ma_30'] = df['value'].rolling(window=30, min_periods=1).mean()
        features['ma_90'] = df['value'].rolling(window=90, min_periods=1).mean()
        
        # Trend indicators
        features['trend_7'] = df['value'] - features['ma_7']
        features['trend_30'] = df['value'] - features['ma_30']
        features['trend_90'] = df['value'] - features['ma_90']
        
        # Rate of change
        features['roc_1'] = df['value'].pct_change(1)
        features['roc_7'] = df['value'].pct_change(7)
        features['roc_30'] = df['value'].pct_change(30)
        
        return features
    
    def _generate_seasonal_features(self, df: pd.DataFrame, context: Dict[str, Any]) -> pd.DataFrame:
        """Generate seasonal features."""
        features = pd.DataFrame(index=df.index)
        
        # Cyclical encoding for time components
        features['day_sin'] = np.sin(2 * np.pi * df.index.day / 31)
        features['day_cos'] = np.cos(2 * np.pi * df.index.day / 31)
        features['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
        features['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
        features['quarter_sin'] = np.sin(2 * np.pi * df.index.quarter / 4)
        features['quarter_cos'] = np.cos(2 * np.pi * df.index.quarter / 4)
        
        return features
    
    def _generate_lag_features(self, df: pd.DataFrame, context: Dict[str, Any]) -> pd.DataFrame:
        """Generate lag features."""
        features = pd.DataFrame(index=df.index)
        
        # Lag features
        for lag in [1, 3, 7, 14, 30]:
            features[f'lag_{lag}'] = df['value'].shift(lag)
        
        # Differencing
        features['diff_1'] = df['value'].diff(1)
        features['diff_7'] = df['value'].diff(7)
        features['diff_30'] = df['value'].diff(30)
        
        return features
    
    def _generate_statistical_features(self, df: pd.DataFrame, context: Dict[str, Any]) -> pd.DataFrame:
        """Generate statistical features."""
        features = pd.DataFrame(index=df.index)
        
        # Rolling statistics
        for window in [7, 30, 90]:
            rolling = df['value'].rolling(window=window, min_periods=1)
            features[f'std_{window}'] = rolling.std()
            features[f'min_{window}'] = rolling.min()
            features[f'max_{window}'] = rolling.max()
            features[f'skew_{window}'] = rolling.skew()
            features[f'kurt_{window}'] = rolling.kurt()
        
        # Volatility measures
        features['volatility_7'] = df['value'].rolling(window=7).std()
        features['volatility_30'] = df['value'].rolling(window=30).std()
        
        return features
    
    def _generate_business_context_features(self, df: pd.DataFrame, context: Dict[str, Any]) -> pd.DataFrame:
        """Generate business context features."""
        features = pd.DataFrame(index=df.index)
        
        # Business calendar features
        business_events = context.get('business_events', [])
        for event in business_events:
            event_name = event.get('name', 'event')
            event_dates = pd.to_datetime(event.get('dates', []))
            
            # Create binary indicator for event dates
            features[f'event_{event_name}'] = df.index.isin(event_dates).astype(int)
            
            # Create proximity features (days until/since event)
            for date in event_dates:
                days_diff = (df.index - date).days
                features[f'days_to_{event_name}'] = np.where(
                    np.abs(days_diff) <= 30, days_diff, 0
                )
        
        # External factors
        if 'external_factors' in context:
            for factor, values in context['external_factors'].items():
                if len(values) == len(df):
                    features[f'external_{factor}'] = values
        
        return features
    
    async def _ml_prediction(
        self,
        metric_name: str,
        features_df: pd.DataFrame,
        horizon_days: int
    ) -> QualityPrediction:
        """Generate ML-based prediction."""
        
        try:
            # Prepare target variable
            target = features_df['value'].values
            feature_cols = [col for col in features_df.columns if col != 'value']
            X = features_df[feature_cols].values
            
            # Handle missing values
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Split data
            train_size = int(len(X_scaled) * self.config.train_test_split_ratio)
            X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
            y_train, y_test = target[:train_size], target[train_size:]
            
            # Train model
            model = self._get_prediction_model()
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred) if len(y_test) > 0 else 0.5
            mae = mean_absolute_error(y_test, y_pred) if len(y_test) > 0 else 0.1
            
            # Generate future prediction
            last_features = X_scaled[-1].reshape(1, -1)
            predicted_value = model.predict(last_features)[0]
            
            # Calculate confidence interval (simplified)
            prediction_std = mae * 1.96  # 95% confidence interval approximation
            confidence_lower = predicted_value - prediction_std
            confidence_upper = predicted_value + prediction_std
            
            # Assess risk
            current_value = target[-1]
            risk_level = self._assess_prediction_risk(predicted_value, current_value, prediction_std)
            
            # Generate recommendations
            recommendations = self._generate_prediction_recommendations(
                metric_name, predicted_value, current_value, risk_level
            )
            
            return QualityPrediction(
                metric_name=metric_name,
                prediction_timestamp=datetime.now(),
                prediction_horizon_days=horizon_days,
                predicted_value=max(0, min(1, predicted_value)),  # Clamp to [0,1]
                confidence_interval_lower=max(0, confidence_lower),
                confidence_interval_upper=min(1, confidence_upper),
                prediction_confidence=min(r2, 0.95),
                model_accuracy=r2,
                historical_variance=np.std(target),
                trend_strength=abs(predicted_value - current_value) / max(current_value, 0.01),
                risk_level=risk_level,
                recommended_actions=recommendations,
                feature_importance=self._get_feature_importance(model, feature_cols)
            )
            
        except Exception as e:
            self.logger.error(f"ML prediction failed for {metric_name}: {str(e)}")
            return await self._statistical_prediction(
                metric_name, features_df['value'], horizon_days
            )
    
    async def _statistical_prediction(
        self,
        metric_name: str,
        metric_series: pd.Series,
        horizon_days: int
    ) -> QualityPrediction:
        """Generate statistical prediction when ML is not available."""
        
        # Simple statistical prediction
        recent_values = metric_series.tail(min(30, len(metric_series)))
        
        # Calculate trend
        if len(recent_values) >= 2:
            x = np.arange(len(recent_values))
            y = recent_values.values
            trend_slope = np.polyfit(x, y, 1)[0]
            predicted_value = recent_values.iloc[-1] + (trend_slope * horizon_days)
        else:
            predicted_value = recent_values.mean()
        
        # Calculate confidence interval
        std_dev = recent_values.std()
        confidence_lower = predicted_value - (1.96 * std_dev)
        confidence_upper = predicted_value + (1.96 * std_dev)
        
        # Assess risk
        current_value = recent_values.iloc[-1]
        risk_level = self._assess_prediction_risk(predicted_value, current_value, std_dev)
        
        # Generate recommendations
        recommendations = self._generate_prediction_recommendations(
            metric_name, predicted_value, current_value, risk_level
        )
        
        return QualityPrediction(
            metric_name=metric_name,
            prediction_timestamp=datetime.now(),
            prediction_horizon_days=horizon_days,
            predicted_value=max(0, min(1, predicted_value)),
            confidence_interval_lower=max(0, confidence_lower),
            confidence_interval_upper=min(1, confidence_upper),
            prediction_confidence=0.7,  # Moderate confidence for statistical prediction
            model_accuracy=0.6,
            historical_variance=std_dev,
            trend_strength=abs(trend_slope) if 'trend_slope' in locals() else 0.0,
            risk_level=risk_level,
            recommended_actions=recommendations
        )
    
    def _get_prediction_model(self):
        """Get the configured prediction model."""
        if self.config.prediction_algorithm == "linear":
            return LinearRegression()
        elif self.config.prediction_algorithm == "random_forest":
            return RandomForestRegressor(n_estimators=100, random_state=42)
        else:  # gradient_boosting
            return GradientBoostingRegressor(n_estimators=100, random_state=42)
    
    def _get_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """Get feature importance from trained model."""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            return dict(zip(feature_names, importance.tolist()))
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_)
            return dict(zip(feature_names, importance.tolist()))
        else:
            return {}
    
    def _assess_prediction_risk(self, predicted: float, current: float, uncertainty: float) -> str:
        """Assess the risk level of a prediction."""
        change = abs(predicted - current)
        
        if change > 0.2 or uncertainty > 0.15:
            return "critical"
        elif change > 0.1 or uncertainty > 0.1:
            return "high"
        elif change > 0.05 or uncertainty > 0.05:
            return "medium"
        else:
            return "low"
    
    def _generate_prediction_recommendations(
        self,
        metric_name: str,
        predicted: float,
        current: float,
        risk_level: str
    ) -> List[str]:
        """Generate recommendations based on prediction."""
        recommendations = []
        
        change = predicted - current
        
        if risk_level in ["critical", "high"]:
            if change < -0.1:
                recommendations.extend([
                    f"Immediate intervention required - {metric_name} declining",
                    f"Review data quality processes for {metric_name}",
                    "Implement monitoring alerts for early detection"
                ])
            elif change > 0.1:
                recommendations.extend([
                    f"Monitor {metric_name} improvement sustainability",
                    "Document successful practices for replication"
                ])
        
        if risk_level == "medium":
            recommendations.append(f"Monitor {metric_name} trends closely")
        
        return recommendations
    
    def _create_default_prediction(self, metric_name: str, horizon_days: int) -> QualityPrediction:
        """Create a default prediction when data is insufficient."""
        return QualityPrediction(
            metric_name=metric_name,
            prediction_timestamp=datetime.now(),
            prediction_horizon_days=horizon_days,
            predicted_value=0.8,  # Assume reasonable quality
            confidence_interval_lower=0.7,
            confidence_interval_upper=0.9,
            prediction_confidence=0.5,  # Low confidence due to insufficient data
            model_accuracy=0.5,
            historical_variance=0.1,
            trend_strength=0.0,
            risk_level="medium",
            recommended_actions=["Collect more historical data for better predictions"]
        )
    
    def _is_prediction_cached(self, cache_key: str) -> bool:
        """Check if prediction is cached and still valid."""
        if cache_key not in self.prediction_cache:
            return False
        
        expiry_time = self.cache_expiry.get(cache_key, datetime.min)
        return datetime.now() < expiry_time
    
    # Investment optimization methods
    
    async def _estimate_investment_requirements(
        self,
        improvement_gaps: Dict[str, float],
        constraints: Dict[str, Any]
    ) -> Dict[str, Dict[str, float]]:
        """Estimate investment requirements for quality improvements."""
        
        requirements = {}
        
        for metric, gap in improvement_gaps.items():
            # Base cost factors per metric type
            base_costs = {
                "completeness": {"technology": 50000, "people": 30000, "process": 20000},
                "accuracy": {"technology": 80000, "people": 40000, "process": 30000},
                "consistency": {"technology": 60000, "people": 35000, "process": 25000},
                "validity": {"technology": 45000, "people": 25000, "process": 15000},
                "uniqueness": {"technology": 35000, "people": 20000, "process": 10000},
                "timeliness": {"technology": 70000, "people": 30000, "process": 20000}
            }
            
            # Scale by improvement gap
            metric_costs = base_costs.get(metric, base_costs["accuracy"])
            requirements[metric] = {
                area: cost * gap * constraints.get("cost_multiplier", 1.0)
                for area, cost in metric_costs.items()
            }
        
        return requirements
    
    async def _optimize_investment_allocation(
        self,
        budget: float,
        requirements: Dict[str, Dict[str, float]],
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize investment allocation across areas and metrics."""
        
        # Calculate total requirements
        total_requirements = {}
        for metric_reqs in requirements.values():
            for area, amount in metric_reqs.items():
                total_requirements[area] = total_requirements.get(area, 0) + amount
        
        total_needed = sum(total_requirements.values())
        
        # Allocate budget proportionally if needed
        if total_needed > budget:
            scale_factor = budget / total_needed
            area_allocations = {
                area: amount * scale_factor
                for area, amount in total_requirements.items()
            }
        else:
            area_allocations = total_requirements.copy()
        
        # Generate priority initiatives based on allocation
        initiatives = []
        if area_allocations.get("technology", 0) > budget * 0.3:
            initiatives.extend([
                "Implement automated data quality monitoring",
                "Deploy ML-powered quality detection"
            ])
        
        if area_allocations.get("people", 0) > budget * 0.3:
            initiatives.extend([
                "Hire dedicated data quality specialists",
                "Expand data steward program"
            ])
        
        if area_allocations.get("process", 0) > budget * 0.2:
            initiatives.extend([
                "Standardize quality processes",
                "Implement quality governance framework"
            ])
        
        return {
            "areas": area_allocations,
            "initiatives": initiatives,
            "timeline_months": max(6, min(24, int(budget / 100000)))  # Rough timeline estimate
        }
    
    async def _calculate_investment_outcomes(
        self,
        allocation: Dict[str, Any],
        improvement_gaps: Dict[str, float]
    ) -> Dict[str, Any]:
        """Calculate expected outcomes from investment."""
        
        total_investment = sum(allocation["areas"].values())
        
        # Estimate quality improvement (diminishing returns)
        quality_improvement = 0
        for gap in improvement_gaps.values():
            # Assume 70% of gap can be closed with sufficient investment
            potential_improvement = gap * 0.7
            # Apply diminishing returns based on investment level
            investment_factor = min(1.0, total_investment / 500000)  # $500k reference
            quality_improvement += potential_improvement * investment_factor
        
        quality_improvement = quality_improvement / max(len(improvement_gaps), 1)
        
        # Estimate ROI based on quality improvement
        # Assume 1% quality improvement = 5% cost savings
        cost_savings = quality_improvement * 5 * total_investment
        roi = (cost_savings / total_investment) * 100 if total_investment > 0 else 0
        
        # Risk reduction (assume quality improvement reduces operational risks)
        risk_reduction = quality_improvement * 0.8
        
        # Payback period (months)
        monthly_savings = cost_savings / 12
        payback_months = int(total_investment / monthly_savings) if monthly_savings > 0 else None
        
        return {
            "quality_improvement": quality_improvement,
            "roi": roi,
            "cost_savings": cost_savings,
            "risk_reduction": risk_reduction,
            "payback_months": payback_months,
            "confidence": 0.75  # Moderate confidence in estimates
        }
    
    def _generate_investment_rationale(
        self,
        allocation: Dict[str, Any],
        outcomes: Dict[str, Any]
    ) -> str:
        """Generate rationale for investment recommendation."""
        
        top_area = max(allocation["areas"].items(), key=lambda x: x[1])
        
        rationale = f"Recommended investment strategy focuses on {top_area[0]} with ${top_area[1]:,.0f} allocation. "
        rationale += f"Expected {outcomes['quality_improvement']*100:.1f}% quality improvement with "
        rationale += f"{outcomes['roi']:.0f}% ROI over {allocation['timeline_months']} months. "
        
        if outcomes.get('payback_months'):
            rationale += f"Investment payback expected in {outcomes['payback_months']} months."
        
        return rationale
    
    # Competitive analysis methods
    
    def _calculate_market_position(
        self,
        org_metrics: Dict[str, float],
        benchmarks: List[IndustryBenchmark]
    ) -> Dict[str, Any]:
        """Calculate organization's market position."""
        
        if not benchmarks:
            return {
                "position": "unknown",
                "score": 0.5,
                "peer_scores": {},
                "leaders": []
            }
        
        # Use first benchmark for analysis
        benchmark = benchmarks[0]
        org_score = org_metrics.get("overall_quality_score", 0.7)
        
        # Determine position based on quartiles
        if org_score >= benchmark.top_quartile_score:
            position = "leader"
        elif org_score >= benchmark.median_quality_score:
            position = "challenger"
        elif org_score >= benchmark.bottom_quartile_score:
            position = "follower"
        else:
            position = "niche"
        
        # Calculate relative score
        score_range = benchmark.top_quartile_score - benchmark.bottom_quartile_score
        relative_score = (org_score - benchmark.bottom_quartile_score) / max(score_range, 0.01)
        
        return {
            "position": position,
            "score": max(0, min(1, relative_score)),
            "peer_scores": {
                "median": benchmark.median_quality_score,
                "top_quartile": benchmark.top_quartile_score,
                "organization": org_score
            },
            "leaders": ["Industry Leader A", "Industry Leader B"]  # Placeholder
        }
    
    def _analyze_maturity_gap(
        self,
        org_metrics: Dict[str, float],
        benchmarks: List[IndustryBenchmark]
    ) -> float:
        """Analyze quality maturity gap."""
        
        if not benchmarks:
            return 0.0
        
        benchmark = benchmarks[0]
        org_score = org_metrics.get("overall_quality_score", 0.7)
        
        # Calculate gap to top quartile
        gap = max(0, benchmark.top_quartile_score - org_score)
        
        return gap
    
    def _calculate_investment_gap(
        self,
        org_metrics: Dict[str, float],
        benchmarks: List[IndustryBenchmark],
        market_context: Dict[str, Any]
    ) -> float:
        """Calculate investment gap compared to industry."""
        
        if not benchmarks:
            return 0.0
        
        benchmark = benchmarks[0]
        org_investment = market_context.get("quality_investment_percentage", 2.0)
        
        # Calculate gap to median investment
        gap = max(0, benchmark.median_quality_investment_percentage - org_investment)
        
        return (gap / benchmark.median_quality_investment_percentage) * 100
    
    def _identify_competitive_advantages_and_gaps(
        self,
        org_metrics: Dict[str, float],
        benchmarks: List[IndustryBenchmark]
    ) -> Tuple[List[str], List[str]]:
        """Identify competitive advantages and gaps."""
        
        advantages = []
        gaps = []
        
        if not benchmarks:
            return advantages, gaps
        
        benchmark = benchmarks[0]
        
        # Analyze each metric
        for metric, value in org_metrics.items():
            if "score" in metric:
                benchmark_value = getattr(benchmark, "median_quality_score", 0.7)
                
                if value > benchmark_value * 1.1:
                    advantages.append(f"Strong {metric.replace('_score', '')} performance")
                elif value < benchmark_value * 0.9:
                    gaps.append(f"Below-average {metric.replace('_score', '')} performance")
        
        # Add general gaps if organization is behind
        org_overall = org_metrics.get("overall_quality_score", 0.7)
        if org_overall < benchmark.median_quality_score:
            gaps.extend([
                "Data governance framework maturity",
                "Automated quality monitoring capabilities",
                "Cross-functional quality collaboration"
            ])
        
        return advantages, gaps
    
    def _assess_competitive_threats(
        self,
        market_position: Dict[str, Any],
        maturity_gap: float,
        market_context: Dict[str, Any]
    ) -> List[str]:
        """Assess competitive threats."""
        
        threats = []
        
        if market_position["position"] in ["follower", "niche"]:
            threats.extend([
                "Risk of customer loss due to quality gaps",
                "Competitive disadvantage in data-driven initiatives",
                "Potential regulatory compliance issues"
            ])
        
        if maturity_gap > 0.2:
            threats.append("Significant quality maturity gap vs industry leaders")
        
        if market_context.get("competitive_pressure") == "high":
            threats.append("High competitive pressure requiring quality excellence")
        
        return threats
    
    def _identify_improvement_opportunities(
        self,
        gaps: List[str],
        market_context: Dict[str, Any]
    ) -> List[str]:
        """Identify improvement opportunities."""
        
        opportunities = []
        
        if gaps:
            opportunities.extend([
                "Implement best-practice quality frameworks",
                "Leverage automation for quality monitoring",
                "Develop center of excellence for data quality"
            ])
        
        if market_context.get("digital_transformation_focus"):
            opportunities.append("Integrate quality into digital transformation initiatives")
        
        if market_context.get("growth_phase"):
            opportunities.append("Build scalable quality infrastructure for growth")
        
        return opportunities