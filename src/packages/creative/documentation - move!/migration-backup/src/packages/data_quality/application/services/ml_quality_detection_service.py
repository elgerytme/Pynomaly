"""ML-Enhanced Quality Detection Service.

Service for detecting quality anomalies using machine learning models
including unsupervised learning, pattern recognition, and predictive analytics.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging
from collections import defaultdict
from enum import Enum
import pickle
import warnings
from pathlib import Path

# ML imports
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
import joblib

from ...domain.entities.quality_anomaly import (
    QualityAnomaly, QualityAnomalyId, AnomalyType, AnomalySeverity, AnomalyStatus,
    AnomalyDetectionResult, AnomalyPattern
)
from ...domain.entities.quality_profile import DataQualityProfile
from ...domain.entities.quality_scores import QualityScores
from .quality_assessment_service import QualityAssessmentService

logger = logging.getLogger(__name__)


class DetectionMethod(Enum):
    """ML detection methods."""
    ISOLATION_FOREST = "isolation_forest"
    LOCAL_OUTLIER_FACTOR = "local_outlier_factor"
    CLUSTERING_ANOMALY = "clustering_anomaly"
    STATISTICAL_OUTLIER = "statistical_outlier"
    ENSEMBLE_DETECTION = "ensemble_detection"
    DRIFT_DETECTION = "drift_detection"
    PATTERN_DEVIATION = "pattern_deviation"
    TEMPORAL_ANOMALY = "temporal_anomaly"


@dataclass(frozen=True)
class MLDetectionConfig:
    """Configuration for ML-enhanced quality detection."""
    # Detection methods to use
    enabled_methods: List[DetectionMethod] = field(default_factory=lambda: [
        DetectionMethod.ISOLATION_FOREST,
        DetectionMethod.LOCAL_OUTLIER_FACTOR,
        DetectionMethod.CLUSTERING_ANOMALY,
        DetectionMethod.STATISTICAL_OUTLIER
    ])
    
    # Model parameters
    contamination_rate: float = 0.1
    n_estimators: int = 100
    random_state: int = 42
    
    # Ensemble parameters
    ensemble_threshold: float = 0.6
    min_ensemble_agreement: int = 2
    
    # Preprocessing
    enable_feature_scaling: bool = True
    enable_dimensionality_reduction: bool = False
    pca_components: int = 10
    
    # Temporal analysis
    enable_temporal_analysis: bool = True
    temporal_window_size: int = 30
    seasonal_period: int = 7
    
    # Pattern detection
    enable_pattern_detection: bool = True
    pattern_sensitivity: float = 0.7
    min_pattern_support: int = 5
    
    # Performance
    max_training_samples: int = 50000
    batch_size: int = 1000
    enable_incremental_learning: bool = False
    
    # Model persistence
    model_cache_dir: str = "models/quality_detection"
    model_retrain_interval_hours: int = 24
    
    def __post_init__(self):
        """Validate configuration."""
        if not 0 < self.contamination_rate < 1:
            raise ValueError("Contamination rate must be between 0 and 1")
        if self.n_estimators <= 0:
            raise ValueError("Number of estimators must be positive")
        if not 0 < self.ensemble_threshold < 1:
            raise ValueError("Ensemble threshold must be between 0 and 1")


@dataclass
class DetectionModel:
    """Container for ML detection model."""
    model: Any
    scaler: Optional[Any] = None
    pca: Optional[Any] = None
    method: DetectionMethod = DetectionMethod.ISOLATION_FOREST
    trained_at: datetime = field(default_factory=datetime.now)
    training_data_shape: Tuple[int, int] = (0, 0)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly scores."""
        if self.scaler:
            X = self.scaler.transform(X)
        if self.pca:
            X = self.pca.transform(X)
        
        if hasattr(self.model, 'decision_function'):
            scores = self.model.decision_function(X)
            # Normalize scores to [0, 1] range
            scores = (scores - scores.min()) / (scores.max() - scores.min())
        elif hasattr(self.model, 'predict_proba'):
            scores = self.model.predict_proba(X)[:, 1]
        else:
            predictions = self.model.predict(X)
            scores = np.where(predictions == -1, 1.0, 0.0)
        
        return scores
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance if available."""
        if hasattr(self.model, 'feature_importances_'):
            return dict(zip(self.feature_names, self.model.feature_importances_))
        return {}


class MLQualityDetectionService:
    """Service for ML-enhanced quality detection."""
    
    def __init__(self, config: MLDetectionConfig = None):
        """Initialize ML quality detection service."""
        self.config = config or MLDetectionConfig()
        self.models: Dict[DetectionMethod, DetectionModel] = {}
        self.quality_assessment_service = QualityAssessmentService()
        
        # Historical data for pattern detection
        self._historical_profiles: List[DataQualityProfile] = []
        self._pattern_cache: Dict[str, List[AnomalyPattern]] = {}
        
        # Performance tracking
        self._detection_stats = {
            'total_detections': 0,
            'anomalies_detected': 0,
            'false_positives': 0,
            'true_positives': 0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0
        }
        
        # Create model cache directory
        Path(self.config.model_cache_dir).mkdir(parents=True, exist_ok=True)
    
    def detect_quality_anomalies(self, 
                                df: pd.DataFrame,
                                dataset_id: str,
                                baseline_profile: Optional[DataQualityProfile] = None,
                                historical_profiles: List[DataQualityProfile] = None) -> List[QualityAnomaly]:
        """Detect quality anomalies using ML models."""
        try:
            logger.info(f"Starting ML quality anomaly detection for dataset {dataset_id}")
            
            # Prepare features for ML models
            features = self._extract_quality_features(df, dataset_id)
            
            # Load or train models
            self._ensure_models_ready(features)
            
            # Run detection with different methods
            anomalies = []
            
            for method in self.config.enabled_methods:
                if method in self.models:
                    method_anomalies = self._detect_with_method(
                        df, features, method, dataset_id, baseline_profile
                    )
                    anomalies.extend(method_anomalies)
            
            # Ensemble detection
            if len(self.config.enabled_methods) > 1:
                ensemble_anomalies = self._ensemble_detection(
                    df, features, dataset_id, baseline_profile
                )
                anomalies.extend(ensemble_anomalies)
            
            # Temporal analysis
            if self.config.enable_temporal_analysis and historical_profiles:
                temporal_anomalies = self._temporal_anomaly_detection(
                    df, dataset_id, historical_profiles
                )
                anomalies.extend(temporal_anomalies)
            
            # Pattern detection
            if self.config.enable_pattern_detection:
                pattern_anomalies = self._pattern_anomaly_detection(
                    df, dataset_id, baseline_profile
                )
                anomalies.extend(pattern_anomalies)
            
            # Remove duplicates and rank by severity
            anomalies = self._deduplicate_and_rank_anomalies(anomalies)
            
            # Update statistics
            self._update_detection_stats(anomalies)
            
            logger.info(f"Detected {len(anomalies)} quality anomalies")
            return anomalies
            
        except Exception as e:
            logger.error(f"ML quality anomaly detection failed: {str(e)}")
            raise
    
    def train_detection_models(self, 
                             training_data: List[Dict[str, Any]],
                             labels: Optional[List[int]] = None) -> None:
        """Train ML detection models."""
        try:
            logger.info("Training ML quality detection models")
            
            # Prepare training features
            X = self._prepare_training_features(training_data)
            
            # Train models for each enabled method
            for method in self.config.enabled_methods:
                logger.info(f"Training {method.value} model")
                
                if method == DetectionMethod.ISOLATION_FOREST:
                    model = self._train_isolation_forest(X)
                elif method == DetectionMethod.LOCAL_OUTLIER_FACTOR:
                    model = self._train_local_outlier_factor(X)
                elif method == DetectionMethod.CLUSTERING_ANOMALY:
                    model = self._train_clustering_anomaly(X)
                elif method == DetectionMethod.STATISTICAL_OUTLIER:
                    model = self._train_statistical_outlier(X)
                elif method == DetectionMethod.ENSEMBLE_DETECTION:
                    model = self._train_ensemble_detection(X, labels)
                else:
                    logger.warning(f"Unknown detection method: {method}")
                    continue
                
                self.models[method] = model
                
                # Save model
                self._save_model(model, method)
            
            logger.info("Model training completed")
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            raise
    
    def predict_quality_degradation(self, 
                                  df: pd.DataFrame,
                                  dataset_id: str,
                                  forecast_horizon: int = 7) -> Dict[str, Any]:
        """Predict future quality degradation."""
        try:
            logger.info(f"Predicting quality degradation for dataset {dataset_id}")
            
            # Extract time series features
            features = self._extract_temporal_features(df, dataset_id)
            
            # Use historical profiles for prediction
            historical_features = self._get_historical_features(dataset_id)
            
            if len(historical_features) < 10:
                logger.warning("Insufficient historical data for prediction")
                return {'prediction': 'insufficient_data'}
            
            # Prepare time series data
            X, y = self._prepare_time_series_data(historical_features)
            
            # Train regression model for prediction
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import mean_squared_error, r2_score
            
            model = LinearRegression()
            model.fit(X[:-forecast_horizon], y[:-forecast_horizon])
            
            # Make predictions
            predictions = model.predict(X[-forecast_horizon:])
            
            # Calculate prediction confidence
            test_predictions = model.predict(X[-forecast_horizon:])
            mse = mean_squared_error(y[-forecast_horizon:], test_predictions)
            r2 = r2_score(y[-forecast_horizon:], test_predictions)
            
            # Generate prediction results
            prediction_results = {
                'dataset_id': dataset_id,
                'forecast_horizon_days': forecast_horizon,
                'predictions': predictions.tolist(),
                'confidence_score': max(0, r2),
                'prediction_error': mse,
                'trend_direction': 'improving' if np.mean(predictions) > np.mean(y[-forecast_horizon:]) else 'degrading',
                'predicted_at': datetime.now().isoformat(),
                'model_performance': {
                    'r2_score': r2,
                    'mse': mse,
                    'training_samples': len(X) - forecast_horizon
                }
            }
            
            return prediction_results
            
        except Exception as e:
            logger.error(f"Quality degradation prediction failed: {str(e)}")
            raise
    
    def analyze_anomaly_patterns(self, 
                               anomalies: List[QualityAnomaly],
                               lookback_days: int = 30) -> Dict[str, Any]:
        """Analyze patterns in detected anomalies."""
        try:
            logger.info("Analyzing anomaly patterns")
            
            # Filter recent anomalies
            cutoff_date = datetime.now() - timedelta(days=lookback_days)
            recent_anomalies = [
                anomaly for anomaly in anomalies 
                if anomaly.detected_at >= cutoff_date
            ]
            
            if not recent_anomalies:
                return {'patterns': [], 'insights': []}
            
            # Analyze patterns
            patterns = self._identify_anomaly_patterns(recent_anomalies)
            
            # Generate insights
            insights = self._generate_pattern_insights(patterns, recent_anomalies)
            
            return {
                'analysis_period_days': lookback_days,
                'total_anomalies': len(recent_anomalies),
                'patterns': patterns,
                'insights': insights,
                'pattern_strength': self._calculate_pattern_strength(patterns),
                'analyzed_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Anomaly pattern analysis failed: {str(e)}")
            raise
    
    def _extract_quality_features(self, df: pd.DataFrame, dataset_id: str) -> np.ndarray:
        """Extract features for ML quality detection."""
        features = []
        
        # Basic statistics
        features.extend([
            len(df),  # Row count
            len(df.columns),  # Column count
            df.isnull().sum().sum(),  # Total missing values
            df.duplicated().sum(),  # Duplicate rows
        ])
        
        # Per-column statistics
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                # Numerical column features
                features.extend([
                    df[col].mean(),
                    df[col].std(),
                    df[col].skew(),
                    df[col].kurtosis(),
                    df[col].quantile(0.25),
                    df[col].quantile(0.75),
                    df[col].nunique(),
                    df[col].isnull().sum(),
                ])
            else:
                # Categorical column features
                features.extend([
                    df[col].nunique(),
                    df[col].isnull().sum(),
                    len(df[col].astype(str).str.len().mean() if not df[col].empty else 0),
                    0, 0, 0, 0, 0  # Padding for numerical features
                ])
        
        # Pad or truncate to fixed size
        max_features = 1000
        if len(features) > max_features:
            features = features[:max_features]
        else:
            features.extend([0] * (max_features - len(features)))
        
        return np.array(features).reshape(1, -1)
    
    def _ensure_models_ready(self, features: np.ndarray) -> None:
        """Ensure ML models are ready for detection."""
        for method in self.config.enabled_methods:
            if method not in self.models:
                # Try to load existing model
                if not self._load_model(method):
                    # Train new model with synthetic data
                    self._train_default_model(method, features)
    
    def _detect_with_method(self, 
                          df: pd.DataFrame,
                          features: np.ndarray,
                          method: DetectionMethod,
                          dataset_id: str,
                          baseline_profile: Optional[DataQualityProfile] = None) -> List[QualityAnomaly]:
        """Detect anomalies using specific method."""
        try:
            model = self.models[method]
            
            # Get anomaly scores
            scores = model.predict(features)
            anomaly_score = scores[0] if len(scores) > 0 else 0.0
            
            # Determine if anomaly
            threshold = 0.5
            if anomaly_score > threshold:
                # Create anomaly detection result
                detection_result = AnomalyDetectionResult(
                    anomaly_score=anomaly_score,
                    confidence_score=min(1.0, anomaly_score * 1.2),
                    feature_contributions=model.get_feature_importance(),
                    detection_method=method.value,
                    model_version=model.training_data_shape[0],
                    detected_at=datetime.now()
                )
                
                # Calculate quality impact
                quality_impact = self._calculate_quality_impact(df, anomaly_score)
                
                # Determine affected columns and records
                affected_columns = self._identify_affected_columns(df, model)
                affected_records = max(1, int(len(df) * (anomaly_score * 0.1)))
                
                # Create anomaly
                anomaly = QualityAnomaly(
                    anomaly_id=QualityAnomalyId(),
                    dataset_id=dataset_id,
                    anomaly_type=self._map_method_to_anomaly_type(method),
                    severity=self._determine_severity(anomaly_score),
                    status=AnomalyStatus.DETECTED,
                    title=f"{method.value.replace('_', ' ').title()} Anomaly Detected",
                    description=self._generate_anomaly_description(method, anomaly_score),
                    detection_result=detection_result,
                    affected_columns=affected_columns,
                    affected_records=affected_records,
                    quality_impact=quality_impact,
                    root_cause_analysis=self._analyze_root_cause(df, method, anomaly_score),
                    model_name=f"{method.value}_detector",
                    model_version="1.0.0"
                )
                
                return [anomaly]
            
            return []
            
        except Exception as e:
            logger.error(f"Detection with method {method} failed: {str(e)}")
            return []
    
    def _ensemble_detection(self, 
                          df: pd.DataFrame,
                          features: np.ndarray,
                          dataset_id: str,
                          baseline_profile: Optional[DataQualityProfile] = None) -> List[QualityAnomaly]:
        """Perform ensemble anomaly detection."""
        try:
            # Collect scores from all models
            method_scores = {}
            for method in self.config.enabled_methods:
                if method in self.models:
                    scores = self.models[method].predict(features)
                    method_scores[method] = scores[0] if len(scores) > 0 else 0.0
            
            if not method_scores:
                return []
            
            # Calculate ensemble score
            ensemble_score = np.mean(list(method_scores.values()))
            
            # Check ensemble agreement
            high_score_methods = [
                method for method, score in method_scores.items()
                if score > 0.5
            ]
            
            if (ensemble_score > self.config.ensemble_threshold and
                len(high_score_methods) >= self.config.min_ensemble_agreement):
                
                # Create ensemble detection result
                detection_result = AnomalyDetectionResult(
                    anomaly_score=ensemble_score,
                    confidence_score=min(1.0, ensemble_score * 1.1),
                    feature_contributions={},
                    detection_method="ensemble_detection",
                    model_version="1.0.0",
                    detected_at=datetime.now()
                )
                
                # Calculate quality impact
                quality_impact = self._calculate_quality_impact(df, ensemble_score)
                
                # Create ensemble anomaly
                anomaly = QualityAnomaly(
                    anomaly_id=QualityAnomalyId(),
                    dataset_id=dataset_id,
                    anomaly_type=AnomalyType.PATTERN_DEVIATION,
                    severity=self._determine_severity(ensemble_score),
                    status=AnomalyStatus.DETECTED,
                    title="Ensemble Anomaly Detection",
                    description=f"Ensemble of {len(high_score_methods)} methods detected anomaly",
                    detection_result=detection_result,
                    affected_columns=list(df.columns),
                    affected_records=max(1, int(len(df) * (ensemble_score * 0.1))),
                    quality_impact=quality_impact,
                    root_cause_analysis={
                        'ensemble_score': ensemble_score,
                        'method_scores': method_scores,
                        'agreeing_methods': [m.value for m in high_score_methods]
                    },
                    model_name="ensemble_detector",
                    model_version="1.0.0"
                )
                
                return [anomaly]
            
            return []
            
        except Exception as e:
            logger.error(f"Ensemble detection failed: {str(e)}")
            return []
    
    def _temporal_anomaly_detection(self, 
                                  df: pd.DataFrame,
                                  dataset_id: str,
                                  historical_profiles: List[DataQualityProfile]) -> List[QualityAnomaly]:
        """Detect temporal anomalies in quality patterns."""
        try:
            if len(historical_profiles) < 5:
                return []
            
            # Extract temporal features
            current_features = self._extract_quality_features(df, dataset_id)
            
            # Get historical features
            historical_features = []
            for profile in historical_profiles[-10:]:  # Last 10 profiles
                # Extract features from profile (simplified)
                features = [
                    profile.quality_scores.overall_score,
                    profile.quality_scores.completeness_score,
                    profile.quality_scores.accuracy_score,
                    profile.quality_scores.consistency_score,
                    profile.quality_scores.validity_score,
                    profile.quality_scores.uniqueness_score,
                    profile.quality_scores.timeliness_score,
                    len(profile.quality_issues),
                    profile.record_count,
                    profile.column_count
                ]
                historical_features.append(features)
            
            # Detect temporal anomalies using statistical methods
            historical_array = np.array(historical_features)
            current_array = np.array(current_features[:len(historical_features[0])])
            
            # Calculate z-scores
            mean_historical = np.mean(historical_array, axis=0)
            std_historical = np.std(historical_array, axis=0)
            
            z_scores = np.abs((current_array - mean_historical) / (std_historical + 1e-8))
            
            # Detect anomalies (z-score > 2.5)
            anomaly_threshold = 2.5
            temporal_anomalies = []
            
            for i, z_score in enumerate(z_scores):
                if z_score > anomaly_threshold:
                    feature_names = [
                        'overall_score', 'completeness_score', 'accuracy_score',
                        'consistency_score', 'validity_score', 'uniqueness_score',
                        'timeliness_score', 'issue_count', 'record_count', 'column_count'
                    ]
                    
                    anomaly_score = min(1.0, z_score / 5.0)
                    
                    detection_result = AnomalyDetectionResult(
                        anomaly_score=anomaly_score,
                        confidence_score=min(1.0, anomaly_score * 1.2),
                        feature_contributions={feature_names[i]: anomaly_score},
                        detection_method="temporal_anomaly_detection",
                        model_version="1.0.0",
                        detected_at=datetime.now()
                    )
                    
                    quality_impact = self._calculate_quality_impact(df, anomaly_score)
                    
                    anomaly = QualityAnomaly(
                        anomaly_id=QualityAnomalyId(),
                        dataset_id=dataset_id,
                        anomaly_type=AnomalyType.TEMPORAL_ANOMALY,
                        severity=self._determine_severity(anomaly_score),
                        status=AnomalyStatus.DETECTED,
                        title=f"Temporal Anomaly in {feature_names[i]}",
                        description=f"Significant deviation detected in {feature_names[i]} (z-score: {z_score:.2f})",
                        detection_result=detection_result,
                        affected_columns=list(df.columns),
                        affected_records=max(1, int(len(df) * 0.1)),
                        quality_impact=quality_impact,
                        root_cause_analysis={
                            'z_score': z_score,
                            'current_value': current_array[i],
                            'historical_mean': mean_historical[i],
                            'historical_std': std_historical[i],
                            'feature_name': feature_names[i]
                        },
                        model_name="temporal_detector",
                        model_version="1.0.0"
                    )
                    
                    temporal_anomalies.append(anomaly)
            
            return temporal_anomalies
            
        except Exception as e:
            logger.error(f"Temporal anomaly detection failed: {str(e)}")
            return []
    
    def _pattern_anomaly_detection(self, 
                                 df: pd.DataFrame,
                                 dataset_id: str,
                                 baseline_profile: Optional[DataQualityProfile] = None) -> List[QualityAnomaly]:
        """Detect pattern anomalies in data quality."""
        try:
            anomalies = []
            
            # Check for unusual pattern in data distribution
            for col in df.select_dtypes(include=['int64', 'float64']).columns:
                if df[col].nunique() > 10:  # Enough unique values
                    # Check for unusual distribution patterns
                    skewness = df[col].skew()
                    kurtosis = df[col].kurtosis()
                    
                    # Detect extreme skewness or kurtosis
                    if abs(skewness) > 3 or abs(kurtosis) > 10:
                        anomaly_score = min(1.0, (abs(skewness) + abs(kurtosis)) / 15)
                        
                        detection_result = AnomalyDetectionResult(
                            anomaly_score=anomaly_score,
                            confidence_score=min(1.0, anomaly_score * 1.1),
                            feature_contributions={col: anomaly_score},
                            detection_method="pattern_deviation_detection",
                            model_version="1.0.0",
                            detected_at=datetime.now()
                        )
                        
                        quality_impact = self._calculate_quality_impact(df, anomaly_score)
                        
                        anomaly = QualityAnomaly(
                            anomaly_id=QualityAnomalyId(),
                            dataset_id=dataset_id,
                            anomaly_type=AnomalyType.PATTERN_DEVIATION,
                            severity=self._determine_severity(anomaly_score),
                            status=AnomalyStatus.DETECTED,
                            title=f"Distribution Pattern Anomaly in {col}",
                            description=f"Unusual distribution pattern detected: skewness={skewness:.2f}, kurtosis={kurtosis:.2f}",
                            detection_result=detection_result,
                            affected_columns=[col],
                            affected_records=len(df),
                            quality_impact=quality_impact,
                            root_cause_analysis={
                                'skewness': skewness,
                                'kurtosis': kurtosis,
                                'column': col,
                                'detection_type': 'distribution_pattern'
                            },
                            model_name="pattern_detector",
                            model_version="1.0.0"
                        )
                        
                        anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Pattern anomaly detection failed: {str(e)}")
            return []
    
    def _train_isolation_forest(self, X: np.ndarray) -> DetectionModel:
        """Train Isolation Forest model."""
        # Prepare data
        scaler = StandardScaler() if self.config.enable_feature_scaling else None
        pca = PCA(n_components=self.config.pca_components) if self.config.enable_dimensionality_reduction else None
        
        X_processed = X.copy()
        if scaler:
            X_processed = scaler.fit_transform(X_processed)
        if pca:
            X_processed = pca.fit_transform(X_processed)
        
        # Train model
        model = IsolationForest(
            contamination=self.config.contamination_rate,
            n_estimators=self.config.n_estimators,
            random_state=self.config.random_state,
            n_jobs=-1
        )
        model.fit(X_processed)
        
        return DetectionModel(
            model=model,
            scaler=scaler,
            pca=pca,
            method=DetectionMethod.ISOLATION_FOREST,
            training_data_shape=X.shape,
            feature_names=[f"feature_{i}" for i in range(X.shape[1])]
        )
    
    def _train_local_outlier_factor(self, X: np.ndarray) -> DetectionModel:
        """Train Local Outlier Factor model."""
        # Prepare data
        scaler = StandardScaler() if self.config.enable_feature_scaling else None
        
        X_processed = X.copy()
        if scaler:
            X_processed = scaler.fit_transform(X_processed)
        
        # Train model
        model = LocalOutlierFactor(
            contamination=self.config.contamination_rate,
            n_neighbors=min(20, max(5, len(X) // 10)),
            n_jobs=-1
        )
        model.fit(X_processed)
        
        return DetectionModel(
            model=model,
            scaler=scaler,
            method=DetectionMethod.LOCAL_OUTLIER_FACTOR,
            training_data_shape=X.shape,
            feature_names=[f"feature_{i}" for i in range(X.shape[1])]
        )
    
    def _train_clustering_anomaly(self, X: np.ndarray) -> DetectionModel:
        """Train clustering-based anomaly detection."""
        # Prepare data
        scaler = StandardScaler() if self.config.enable_feature_scaling else None
        
        X_processed = X.copy()
        if scaler:
            X_processed = scaler.fit_transform(X_processed)
        
        # Train clustering model
        n_clusters = min(8, max(2, len(X) // 100))
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.config.random_state)
        kmeans.fit(X_processed)
        
        return DetectionModel(
            model=kmeans,
            scaler=scaler,
            method=DetectionMethod.CLUSTERING_ANOMALY,
            training_data_shape=X.shape,
            feature_names=[f"feature_{i}" for i in range(X.shape[1])]
        )
    
    def _train_statistical_outlier(self, X: np.ndarray) -> DetectionModel:
        """Train statistical outlier detection."""
        # Statistical model based on z-scores
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0) + 1e-8
        
        model = {'mean': mean, 'std': std, 'threshold': 2.5}
        
        return DetectionModel(
            model=model,
            method=DetectionMethod.STATISTICAL_OUTLIER,
            training_data_shape=X.shape,
            feature_names=[f"feature_{i}" for i in range(X.shape[1])]
        )
    
    def _train_ensemble_detection(self, X: np.ndarray, labels: Optional[List[int]] = None) -> DetectionModel:
        """Train ensemble detection model."""
        if labels is None:
            # Use unsupervised ensemble
            isolation_model = self._train_isolation_forest(X)
            lof_model = self._train_local_outlier_factor(X)
            
            # Combine models
            ensemble_model = {
                'isolation_forest': isolation_model,
                'local_outlier_factor': lof_model
            }
        else:
            # Use supervised ensemble
            scaler = StandardScaler() if self.config.enable_feature_scaling else None
            X_processed = X.copy()
            if scaler:
                X_processed = scaler.fit_transform(X_processed)
            
            # Train Random Forest
            rf_model = RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                random_state=self.config.random_state,
                n_jobs=-1
            )
            rf_model.fit(X_processed, labels)
            
            ensemble_model = rf_model
        
        return DetectionModel(
            model=ensemble_model,
            method=DetectionMethod.ENSEMBLE_DETECTION,
            training_data_shape=X.shape,
            feature_names=[f"feature_{i}" for i in range(X.shape[1])]
        )
    
    def _train_default_model(self, method: DetectionMethod, features: np.ndarray) -> None:
        """Train default model with synthetic data."""
        # Generate synthetic training data
        n_samples = 1000
        n_features = features.shape[1]
        
        # Generate normal data
        normal_data = np.random.randn(n_samples, n_features)
        
        # Generate anomalous data
        anomaly_data = np.random.randn(int(n_samples * 0.1), n_features) * 3
        
        # Combine data
        X = np.vstack([normal_data, anomaly_data])
        
        # Train model
        if method == DetectionMethod.ISOLATION_FOREST:
            model = self._train_isolation_forest(X)
        elif method == DetectionMethod.LOCAL_OUTLIER_FACTOR:
            model = self._train_local_outlier_factor(X)
        elif method == DetectionMethod.CLUSTERING_ANOMALY:
            model = self._train_clustering_anomaly(X)
        elif method == DetectionMethod.STATISTICAL_OUTLIER:
            model = self._train_statistical_outlier(X)
        else:
            return
        
        self.models[method] = model
    
    def _prepare_training_features(self, training_data: List[Dict[str, Any]]) -> np.ndarray:
        """Prepare features from training data."""
        features = []
        
        for data_point in training_data:
            # Extract features from data point
            point_features = []
            
            # Add basic statistics
            point_features.extend([
                data_point.get('record_count', 0),
                data_point.get('column_count', 0),
                data_point.get('null_count', 0),
                data_point.get('duplicate_count', 0),
                data_point.get('overall_quality_score', 0.0),
                data_point.get('completeness_score', 0.0),
                data_point.get('accuracy_score', 0.0),
                data_point.get('consistency_score', 0.0),
                data_point.get('validity_score', 0.0),
                data_point.get('uniqueness_score', 0.0),
                data_point.get('timeliness_score', 0.0),
                data_point.get('issue_count', 0),
            ])
            
            # Pad or truncate to fixed size
            max_features = 100
            if len(point_features) > max_features:
                point_features = point_features[:max_features]
            else:
                point_features.extend([0] * (max_features - len(point_features)))
            
            features.append(point_features)
        
        return np.array(features)
    
    def _extract_temporal_features(self, df: pd.DataFrame, dataset_id: str) -> List[float]:
        """Extract temporal features for time series analysis."""
        # Get current quality assessment
        profile = self.quality_assessment_service.assess_dataset_quality(df, dataset_id=dataset_id)
        
        return [
            profile.quality_scores.overall_score,
            profile.quality_scores.completeness_score,
            profile.quality_scores.accuracy_score,
            profile.quality_scores.consistency_score,
            profile.quality_scores.validity_score,
            profile.quality_scores.uniqueness_score,
            profile.quality_scores.timeliness_score,
            len(profile.quality_issues),
            profile.record_count,
            profile.column_count
        ]
    
    def _get_historical_features(self, dataset_id: str) -> List[List[float]]:
        """Get historical features for time series analysis."""
        # In a real implementation, this would fetch from a database
        # For now, return empty list
        return []
    
    def _prepare_time_series_data(self, features: List[List[float]]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare time series data for prediction."""
        if len(features) < 2:
            return np.array([]), np.array([])
        
        # Use sliding window approach
        window_size = 5
        X, y = [], []
        
        for i in range(len(features) - window_size):
            X.append(features[i:i+window_size])
            y.append(features[i+window_size][0])  # Predict overall quality score
        
        return np.array(X), np.array(y)
    
    def _calculate_quality_impact(self, df: pd.DataFrame, anomaly_score: float) -> QualityScores:
        """Calculate quality impact of anomaly."""
        # Get baseline quality assessment
        profile = self.quality_assessment_service.assess_dataset_quality(df)
        
        # Adjust scores based on anomaly severity
        impact_factor = anomaly_score * 0.3  # Reduce scores by up to 30%
        
        return QualityScores(
            overall_score=max(0, profile.quality_scores.overall_score - impact_factor),
            completeness_score=max(0, profile.quality_scores.completeness_score - impact_factor),
            accuracy_score=max(0, profile.quality_scores.accuracy_score - impact_factor),
            consistency_score=max(0, profile.quality_scores.consistency_score - impact_factor),
            validity_score=max(0, profile.quality_scores.validity_score - impact_factor),
            uniqueness_score=max(0, profile.quality_scores.uniqueness_score - impact_factor),
            timeliness_score=max(0, profile.quality_scores.timeliness_score - impact_factor),
            scoring_method=profile.quality_scores.scoring_method,
            weight_configuration=profile.quality_scores.weight_configuration
        )
    
    def _identify_affected_columns(self, df: pd.DataFrame, model: DetectionModel) -> List[str]:
        """Identify columns most affected by the anomaly."""
        # Get feature importance
        feature_importance = model.get_feature_importance()
        
        if not feature_importance:
            return list(df.columns)[:5]  # Return first 5 columns as fallback
        
        # Map feature importance to columns (simplified)
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Return top affected columns
        return [col for col, _ in sorted_features[:5] if col in df.columns]
    
    def _map_method_to_anomaly_type(self, method: DetectionMethod) -> AnomalyType:
        """Map detection method to anomaly type."""
        mapping = {
            DetectionMethod.ISOLATION_FOREST: AnomalyType.STATISTICAL_OUTLIER,
            DetectionMethod.LOCAL_OUTLIER_FACTOR: AnomalyType.STATISTICAL_OUTLIER,
            DetectionMethod.CLUSTERING_ANOMALY: AnomalyType.CLUSTERING_ANOMALY,
            DetectionMethod.STATISTICAL_OUTLIER: AnomalyType.STATISTICAL_OUTLIER,
            DetectionMethod.ENSEMBLE_DETECTION: AnomalyType.PATTERN_DEVIATION,
            DetectionMethod.DRIFT_DETECTION: AnomalyType.DRIFT_DETECTION,
            DetectionMethod.PATTERN_DEVIATION: AnomalyType.PATTERN_DEVIATION,
            DetectionMethod.TEMPORAL_ANOMALY: AnomalyType.TEMPORAL_ANOMALY
        }
        return mapping.get(method, AnomalyType.STATISTICAL_OUTLIER)
    
    def _determine_severity(self, anomaly_score: float) -> AnomalySeverity:
        """Determine severity based on anomaly score."""
        if anomaly_score >= 0.8:
            return AnomalySeverity.CRITICAL
        elif anomaly_score >= 0.6:
            return AnomalySeverity.HIGH
        elif anomaly_score >= 0.4:
            return AnomalySeverity.MEDIUM
        else:
            return AnomalySeverity.LOW
    
    def _generate_anomaly_description(self, method: DetectionMethod, anomaly_score: float) -> str:
        """Generate description for anomaly."""
        descriptions = {
            DetectionMethod.ISOLATION_FOREST: f"Isolation Forest detected outlier with score {anomaly_score:.3f}",
            DetectionMethod.LOCAL_OUTLIER_FACTOR: f"Local Outlier Factor detected anomaly with score {anomaly_score:.3f}",
            DetectionMethod.CLUSTERING_ANOMALY: f"Clustering analysis detected anomaly with score {anomaly_score:.3f}",
            DetectionMethod.STATISTICAL_OUTLIER: f"Statistical analysis detected outlier with score {anomaly_score:.3f}",
            DetectionMethod.ENSEMBLE_DETECTION: f"Ensemble of models detected anomaly with score {anomaly_score:.3f}",
            DetectionMethod.DRIFT_DETECTION: f"Data drift detected with score {anomaly_score:.3f}",
            DetectionMethod.PATTERN_DEVIATION: f"Pattern deviation detected with score {anomaly_score:.3f}",
            DetectionMethod.TEMPORAL_ANOMALY: f"Temporal anomaly detected with score {anomaly_score:.3f}"
        }
        return descriptions.get(method, f"Anomaly detected with score {anomaly_score:.3f}")
    
    def _analyze_root_cause(self, df: pd.DataFrame, method: DetectionMethod, anomaly_score: float) -> Dict[str, Any]:
        """Analyze root cause of anomaly."""
        return {
            'detection_method': method.value,
            'anomaly_score': anomaly_score,
            'dataset_shape': df.shape,
            'null_percentage': df.isnull().sum().sum() / df.size * 100,
            'duplicate_percentage': df.duplicated().sum() / len(df) * 100,
            'data_types': df.dtypes.value_counts().to_dict(),
            'potential_causes': self._identify_potential_causes(df, anomaly_score)
        }
    
    def _identify_potential_causes(self, df: pd.DataFrame, anomaly_score: float) -> List[str]:
        """Identify potential causes of anomaly."""
        causes = []
        
        # High null percentage
        null_pct = df.isnull().sum().sum() / df.size * 100
        if null_pct > 10:
            causes.append(f"High missing data percentage: {null_pct:.1f}%")
        
        # High duplicate percentage
        dup_pct = df.duplicated().sum() / len(df) * 100
        if dup_pct > 5:
            causes.append(f"High duplicate percentage: {dup_pct:.1f}%")
        
        # Unusual data types
        if len(df.select_dtypes(include=['object']).columns) > len(df.columns) * 0.8:
            causes.append("High proportion of text columns")
        
        # Very small or large dataset
        if len(df) < 100:
            causes.append("Very small dataset size")
        elif len(df) > 1000000:
            causes.append("Very large dataset size")
        
        return causes
    
    def _deduplicate_and_rank_anomalies(self, anomalies: List[QualityAnomaly]) -> List[QualityAnomaly]:
        """Remove duplicates and rank anomalies by severity."""
        # Remove duplicates based on similarity
        unique_anomalies = []
        seen_signatures = set()
        
        for anomaly in anomalies:
            # Create signature for deduplication
            signature = (
                anomaly.anomaly_type.value,
                frozenset(anomaly.affected_columns),
                round(anomaly.detection_result.anomaly_score, 2)
            )
            
            if signature not in seen_signatures:
                unique_anomalies.append(anomaly)
                seen_signatures.add(signature)
        
        # Sort by severity and anomaly score
        unique_anomalies.sort(
            key=lambda x: (x.get_severity_score(), x.detection_result.anomaly_score),
            reverse=True
        )
        
        return unique_anomalies
    
    def _identify_anomaly_patterns(self, anomalies: List[QualityAnomaly]) -> List[Dict[str, Any]]:
        """Identify patterns in anomalies."""
        patterns = []
        
        # Group by anomaly type
        type_groups = defaultdict(list)
        for anomaly in anomalies:
            type_groups[anomaly.anomaly_type].append(anomaly)
        
        # Analyze each type
        for anomaly_type, type_anomalies in type_groups.items():
            if len(type_anomalies) > 1:
                patterns.append({
                    'pattern_type': 'recurring_anomaly_type',
                    'anomaly_type': anomaly_type.value,
                    'frequency': len(type_anomalies),
                    'avg_severity': np.mean([a.get_severity_score() for a in type_anomalies]),
                    'datasets_affected': len(set(a.dataset_id for a in type_anomalies))
                })
        
        # Time-based patterns
        if len(anomalies) > 5:
            # Group by hour of day
            hour_groups = defaultdict(int)
            for anomaly in anomalies:
                hour_groups[anomaly.detected_at.hour] += 1
            
            # Find peak hours
            peak_hours = [hour for hour, count in hour_groups.items() if count > 1]
            if peak_hours:
                patterns.append({
                    'pattern_type': 'temporal_clustering',
                    'peak_hours': peak_hours,
                    'description': f"Anomalies tend to occur during hours: {peak_hours}"
                })
        
        return patterns
    
    def _generate_pattern_insights(self, patterns: List[Dict[str, Any]], anomalies: List[QualityAnomaly]) -> List[str]:
        """Generate insights from patterns."""
        insights = []
        
        for pattern in patterns:
            if pattern['pattern_type'] == 'recurring_anomaly_type':
                insights.append(
                    f"Recurring {pattern['anomaly_type']} anomalies detected "
                    f"({pattern['frequency']} occurrences)"
                )
            elif pattern['pattern_type'] == 'temporal_clustering':
                insights.append(
                    f"Anomalies cluster around specific time periods: "
                    f"hours {pattern['peak_hours']}"
                )
        
        # General insights
        if len(anomalies) > 0:
            critical_count = sum(1 for a in anomalies if a.is_critical())
            if critical_count > 0:
                insights.append(f"{critical_count} critical anomalies require immediate attention")
            
            recurring_count = sum(1 for a in anomalies if a.is_recurring())
            if recurring_count > 0:
                insights.append(f"{recurring_count} recurring anomalies indicate systemic issues")
        
        return insights
    
    def _calculate_pattern_strength(self, patterns: List[Dict[str, Any]]) -> float:
        """Calculate overall pattern strength."""
        if not patterns:
            return 0.0
        
        total_strength = 0.0
        for pattern in patterns:
            if pattern['pattern_type'] == 'recurring_anomaly_type':
                total_strength += pattern['frequency'] * 0.1
            elif pattern['pattern_type'] == 'temporal_clustering':
                total_strength += len(pattern['peak_hours']) * 0.05
        
        return min(1.0, total_strength)
    
    def _save_model(self, model: DetectionModel, method: DetectionMethod) -> None:
        """Save model to disk."""
        try:
            model_path = Path(self.config.model_cache_dir) / f"{method.value}_model.pkl"
            joblib.dump(model, model_path)
            logger.info(f"Model saved: {model_path}")
        except Exception as e:
            logger.error(f"Failed to save model {method.value}: {str(e)}")
    
    def _load_model(self, method: DetectionMethod) -> bool:
        """Load model from disk."""
        try:
            model_path = Path(self.config.model_cache_dir) / f"{method.value}_model.pkl"
            if model_path.exists():
                model = joblib.load(model_path)
                
                # Check if model is still valid
                if (datetime.now() - model.trained_at).total_seconds() < self.config.model_retrain_interval_hours * 3600:
                    self.models[method] = model
                    logger.info(f"Model loaded: {model_path}")
                    return True
                else:
                    logger.info(f"Model expired: {model_path}")
                    return False
            return False
        except Exception as e:
            logger.error(f"Failed to load model {method.value}: {str(e)}")
            return False
    
    def _update_detection_stats(self, anomalies: List[QualityAnomaly]) -> None:
        """Update detection statistics."""
        self._detection_stats['total_detections'] += 1
        self._detection_stats['anomalies_detected'] += len(anomalies)
        
        # Update other stats (would be updated based on user feedback in real implementation)
        # For now, estimate based on anomaly scores
        for anomaly in anomalies:
            if anomaly.detection_result.anomaly_score > 0.8:
                self._detection_stats['true_positives'] += 1
            elif anomaly.detection_result.anomaly_score < 0.4:
                self._detection_stats['false_positives'] += 1
        
        # Calculate rates
        total_predictions = self._detection_stats['true_positives'] + self._detection_stats['false_positives']
        if total_predictions > 0:
            self._detection_stats['precision'] = self._detection_stats['true_positives'] / total_predictions
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """Get detection service statistics."""
        return self._detection_stats.copy()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        model_info = {}
        
        for method, model in self.models.items():
            model_info[method.value] = {
                'trained_at': model.trained_at.isoformat(),
                'training_data_shape': model.training_data_shape,
                'performance_metrics': model.performance_metrics,
                'feature_count': len(model.feature_names)
            }
        
        return model_info