"""
Comprehensive ML Model Monitoring

Provides specialized monitoring for ML models including performance tracking,
drift detection, bias monitoring, and automated health assessments.
"""

import asyncio
import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import pickle
from pathlib import Path

from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
import shap

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """ML model types."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"
    TIME_SERIES = "time_series"
    NLP = "nlp"
    COMPUTER_VISION = "computer_vision"


class PerformanceMetric(Enum):
    """Model performance metrics."""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    AUC_ROC = "auc_roc"
    MAE = "mae"
    MSE = "mse"
    RMSE = "rmse"
    R2_SCORE = "r2_score"


class DriftType(Enum):
    """Types of drift that can be detected."""
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    PREDICTION_DRIFT = "prediction_drift"


@dataclass
class ModelMetrics:
    """Model performance metrics."""
    model_id: str
    model_version: str
    timestamp: datetime
    predictions_count: int
    latency_p50: float
    latency_p95: float
    latency_p99: float
    throughput: float
    error_rate: float
    performance_metrics: Dict[str, float]
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class DriftResult:
    """Drift detection result."""
    model_id: str
    drift_type: DriftType
    is_drift_detected: bool
    drift_score: float
    confidence: float
    affected_features: List[str]
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BiasResult:
    """Bias detection result."""
    model_id: str
    protected_attribute: str
    bias_detected: bool
    bias_score: float
    fairness_metrics: Dict[str, float]
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelHealth:
    """Overall model health assessment."""
    model_id: str
    health_score: float  # 0-1 scale
    status: str  # healthy, degraded, unhealthy
    performance_health: float
    drift_health: float
    bias_health: float
    operational_health: float
    recommendations: List[str]
    timestamp: datetime


class ModelMonitor:
    """
    Comprehensive ML model monitoring system that tracks performance,
    detects drift and bias, and provides health assessments.
    """
    
    def __init__(
        self,
        storage_path: str = "./model_monitoring",
        drift_detection_window: int = 1000,
        bias_check_interval: timedelta = timedelta(hours=24),
        enable_explainability: bool = True
    ):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.drift_detection_window = drift_detection_window
        self.bias_check_interval = bias_check_interval
        self.enable_explainability = enable_explainability
        
        # Model registry and metadata
        self.models: Dict[str, Dict[str, Any]] = {}
        self.reference_data: Dict[str, pd.DataFrame] = {}
        self.prediction_history: Dict[str, List[Dict[str, Any]]] = {}
        self.performance_history: Dict[str, List[ModelMetrics]] = {}
        self.drift_history: Dict[str, List[DriftResult]] = {}
        self.bias_history: Dict[str, List[BiasResult]] = {}
        
        # Monitoring rules and thresholds
        self.performance_thresholds: Dict[str, Dict[str, float]] = {}
        self.drift_thresholds: Dict[str, float] = {}
        self.bias_thresholds: Dict[str, float] = {}
        
        # Background monitoring
        self.monitoring_tasks: Dict[str, asyncio.Task] = {}
        self.running = False
        
        # Initialize SHAP explainer cache
        self.explainers: Dict[str, Any] = {}
    
    async def start_monitoring(self) -> None:
        """Start background monitoring tasks."""
        self.running = True
        logger.info("Model monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop background monitoring tasks."""
        self.running = False
        
        # Cancel all monitoring tasks
        for task in self.monitoring_tasks.values():
            task.cancel()
        
        self.monitoring_tasks.clear()
        logger.info("Model monitoring stopped")
    
    def register_model(
        self,
        model_id: str,
        model_version: str,
        model_type: ModelType,
        model_artifact: Any = None,
        reference_data: Optional[pd.DataFrame] = None,
        feature_names: Optional[List[str]] = None,
        target_column: Optional[str] = None,
        protected_attributes: Optional[List[str]] = None,
        performance_thresholds: Optional[Dict[str, float]] = None
    ) -> None:
        """Register a model for monitoring."""
        model_key = f"{model_id}:{model_version}"
        
        self.models[model_key] = {
            "model_id": model_id,
            "model_version": model_version,
            "model_type": model_type,
            "model_artifact": model_artifact,
            "feature_names": feature_names or [],
            "target_column": target_column,
            "protected_attributes": protected_attributes or [],
            "registered_at": datetime.utcnow()
        }
        
        # Store reference data for drift detection
        if reference_data is not None:
            self.reference_data[model_key] = reference_data.copy()
        
        # Set performance thresholds
        if performance_thresholds:
            self.performance_thresholds[model_key] = performance_thresholds
        
        # Initialize monitoring data structures
        self.prediction_history[model_key] = []
        self.performance_history[model_key] = []
        self.drift_history[model_key] = []
        self.bias_history[model_key] = []
        
        # Setup SHAP explainer if enabled
        if self.enable_explainability and model_artifact is not None:
            self._setup_explainer(model_key, model_artifact, reference_data)
        
        # Start background monitoring task
        self.monitoring_tasks[model_key] = asyncio.create_task(
            self._monitor_model(model_key)
        )
        
        logger.info(f"Registered model: {model_id}:{model_version}")
    
    def _setup_explainer(self, model_key: str, model_artifact: Any, reference_data: Optional[pd.DataFrame]) -> None:
        """Setup SHAP explainer for the model."""
        try:
            if reference_data is not None and len(reference_data) > 0:
                # Use a sample of reference data for explainer
                sample_size = min(100, len(reference_data))
                background_data = reference_data.sample(n=sample_size, random_state=42)
                
                # Create explainer based on model type
                if hasattr(model_artifact, "predict_proba"):
                    # Classification model
                    self.explainers[model_key] = shap.Explainer(
                        model_artifact.predict_proba,
                        background_data
                    )
                else:
                    # Regression or other models
                    self.explainers[model_key] = shap.Explainer(
                        model_artifact.predict,
                        background_data
                    )
                
                logger.info(f"SHAP explainer setup for model: {model_key}")
                
        except Exception as e:
            logger.warning(f"Failed to setup SHAP explainer for {model_key}: {e}")
    
    async def log_prediction(
        self,
        model_id: str,
        model_version: str,
        input_data: Union[Dict[str, Any], pd.DataFrame],
        prediction: Any,
        prediction_proba: Optional[np.ndarray] = None,
        ground_truth: Optional[Any] = None,
        latency: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log a model prediction for monitoring."""
        model_key = f"{model_id}:{model_version}"
        
        if model_key not in self.models:
            logger.warning(f"Model {model_key} not registered for monitoring")
            return
        
        # Prepare prediction record
        prediction_record = {
            "timestamp": datetime.utcnow(),
            "input_data": input_data,
            "prediction": prediction,
            "prediction_proba": prediction_proba.tolist() if prediction_proba is not None else None,
            "ground_truth": ground_truth,
            "latency": latency,
            "metadata": metadata or {}
        }
        
        # Add feature importance if explainer is available
        if self.enable_explainability and model_key in self.explainers:
            try:
                feature_importance = self._get_feature_importance(model_key, input_data)
                prediction_record["feature_importance"] = feature_importance
            except Exception as e:
                logger.warning(f"Failed to get feature importance: {e}")
        
        # Store prediction
        self.prediction_history[model_key].append(prediction_record)
        
        # Keep only recent predictions (for memory management)
        if len(self.prediction_history[model_key]) > 10000:
            self.prediction_history[model_key] = self.prediction_history[model_key][-5000:]
        
        # Trigger immediate checks if needed
        await self._check_real_time_metrics(model_key)
    
    def _get_feature_importance(self, model_key: str, input_data: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, float]:
        """Get feature importance for a prediction using SHAP."""
        explainer = self.explainers.get(model_key)
        if not explainer:
            return {}
        
        try:
            # Convert input data to DataFrame if needed
            if isinstance(input_data, dict):
                input_df = pd.DataFrame([input_data])
            else:
                input_df = input_data
            
            # Get SHAP values
            shap_values = explainer(input_df)
            
            # Extract feature importance (absolute SHAP values)
            if hasattr(shap_values, 'values'):
                importance_values = np.abs(shap_values.values[0])
                feature_names = self.models[model_key]["feature_names"]
                
                if len(feature_names) == len(importance_values):
                    return dict(zip(feature_names, importance_values.tolist()))
            
        except Exception as e:
            logger.warning(f"Error calculating feature importance: {e}")
        
        return {}
    
    async def _check_real_time_metrics(self, model_key: str) -> None:
        """Check real-time metrics for immediate issues."""
        predictions = self.prediction_history[model_key]
        
        if len(predictions) < 10:  # Need minimum predictions
            return
        
        # Check recent latency
        recent_predictions = predictions[-10:]
        latencies = [p["latency"] for p in recent_predictions if p["latency"] is not None]
        
        if latencies:
            avg_latency = np.mean(latencies)
            p95_latency = np.percentile(latencies, 95)
            
            # Check against thresholds
            thresholds = self.performance_thresholds.get(model_key, {})
            latency_threshold = thresholds.get("latency_p95", 1.0)  # Default 1 second
            
            if p95_latency > latency_threshold:
                logger.warning(f"High latency detected for {model_key}: {p95_latency:.3f}s")
    
    async def _monitor_model(self, model_key: str) -> None:
        """Background monitoring task for a specific model."""
        while self.running:
            try:
                # Calculate performance metrics
                await self._calculate_performance_metrics(model_key)
                
                # Check for drift
                await self._check_drift(model_key)
                
                # Check for bias
                await self._check_bias(model_key)
                
                # Update model health
                await self._update_model_health(model_key)
                
                # Wait before next check
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in model monitoring for {model_key}: {e}")
                await asyncio.sleep(300)
    
    async def _calculate_performance_metrics(self, model_key: str) -> None:
        """Calculate and store performance metrics."""
        predictions = self.prediction_history[model_key]
        
        if len(predictions) < 10:
            return
        
        # Get recent predictions with ground truth
        recent_window = datetime.utcnow() - timedelta(hours=1)
        recent_predictions = [
            p for p in predictions
            if p["timestamp"] >= recent_window and p["ground_truth"] is not None
        ]
        
        if len(recent_predictions) < 5:
            return
        
        model_info = self.models[model_key]
        model_type = model_info["model_type"]
        
        # Extract predictions and ground truth
        y_true = [p["ground_truth"] for p in recent_predictions]
        y_pred = [p["prediction"] for p in recent_predictions]
        y_proba = [p["prediction_proba"] for p in recent_predictions if p["prediction_proba"] is not None]
        
        # Calculate latency metrics
        latencies = [p["latency"] for p in predictions[-100:] if p["latency"] is not None]
        latency_metrics = {}
        if latencies:
            latency_metrics = {
                "latency_p50": np.percentile(latencies, 50),
                "latency_p95": np.percentile(latencies, 95),
                "latency_p99": np.percentile(latencies, 99)
            }
        
        # Calculate throughput and error rate
        throughput = len(predictions[-100:]) / min(100, len(predictions))  # predictions per minute (approx)
        error_count = sum(1 for p in predictions[-100:] if p.get("error", False))
        error_rate = error_count / len(predictions[-100:]) if predictions else 0
        
        # Calculate performance metrics based on model type
        performance_metrics = {}
        
        if model_type == ModelType.CLASSIFICATION:
            try:
                performance_metrics["accuracy"] = accuracy_score(y_true, y_pred)
                performance_metrics["precision"] = precision_score(y_true, y_pred, average="weighted", zero_division=0)
                performance_metrics["recall"] = recall_score(y_true, y_pred, average="weighted", zero_division=0)
                performance_metrics["f1_score"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
                
                # AUC-ROC if probabilities available
                if y_proba and len(set(y_true)) == 2:  # Binary classification
                    y_proba_positive = [p[1] if len(p) > 1 else p[0] for p in y_proba]
                    performance_metrics["auc_roc"] = roc_auc_score(y_true, y_proba_positive)
                    
            except Exception as e:
                logger.warning(f"Error calculating classification metrics: {e}")
        
        elif model_type == ModelType.REGRESSION:
            try:
                from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                
                performance_metrics["mae"] = mean_absolute_error(y_true, y_pred)
                performance_metrics["mse"] = mean_squared_error(y_true, y_pred)
                performance_metrics["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
                performance_metrics["r2_score"] = r2_score(y_true, y_pred)
                
            except Exception as e:
                logger.warning(f"Error calculating regression metrics: {e}")
        
        # Create metrics object
        metrics = ModelMetrics(
            model_id=model_info["model_id"],
            model_version=model_info["model_version"],
            timestamp=datetime.utcnow(),
            predictions_count=len(recent_predictions),
            latency_p50=latency_metrics.get("latency_p50", 0),
            latency_p95=latency_metrics.get("latency_p95", 0),
            latency_p99=latency_metrics.get("latency_p99", 0),
            throughput=throughput,
            error_rate=error_rate,
            performance_metrics=performance_metrics
        )
        
        # Store metrics
        self.performance_history[model_key].append(metrics)
        
        # Keep only recent metrics
        if len(self.performance_history[model_key]) > 1000:
            self.performance_history[model_key] = self.performance_history[model_key][-500:]
    
    async def _check_drift(self, model_key: str) -> None:
        """Check for various types of drift."""
        predictions = self.prediction_history[model_key]
        reference_data = self.reference_data.get(model_key)
        
        if not reference_data or len(predictions) < self.drift_detection_window:
            return
        
        # Get recent data
        recent_predictions = predictions[-self.drift_detection_window:]
        
        # Extract input features for drift detection
        try:
            recent_inputs = []
            for pred in recent_predictions:
                if isinstance(pred["input_data"], dict):
                    recent_inputs.append(pred["input_data"])
                elif isinstance(pred["input_data"], pd.DataFrame):
                    recent_inputs.extend(pred["input_data"].to_dict("records"))
            
            if not recent_inputs:
                return
            
            recent_data = pd.DataFrame(recent_inputs)
            
            # Detect data drift
            data_drift_result = await self._detect_data_drift(model_key, reference_data, recent_data)
            if data_drift_result:
                self.drift_history[model_key].append(data_drift_result)
            
            # Detect prediction drift
            prediction_drift_result = await self._detect_prediction_drift(model_key, recent_predictions)
            if prediction_drift_result:
                self.drift_history[model_key].append(prediction_drift_result)
                
        except Exception as e:
            logger.warning(f"Error in drift detection for {model_key}: {e}")
    
    async def _detect_data_drift(self, model_key: str, reference_data: pd.DataFrame, recent_data: pd.DataFrame) -> Optional[DriftResult]:
        """Detect data drift using statistical tests."""
        drift_threshold = self.drift_thresholds.get(model_key, 0.05)  # Default p-value threshold
        
        common_columns = list(set(reference_data.columns) & set(recent_data.columns))
        if not common_columns:
            return None
        
        drift_scores = []
        affected_features = []
        
        for column in common_columns:
            try:
                ref_values = reference_data[column].dropna()
                recent_values = recent_data[column].dropna()
                
                if len(ref_values) < 10 or len(recent_values) < 10:
                    continue
                
                # Choose test based on data type
                if pd.api.types.is_numeric_dtype(ref_values):
                    # Kolmogorov-Smirnov test for numerical data
                    statistic, p_value = stats.ks_2samp(ref_values, recent_values)
                else:
                    # Chi-square test for categorical data
                    ref_counts = ref_values.value_counts()
                    recent_counts = recent_values.value_counts()
                    
                    # Align categories
                    all_categories = set(ref_counts.index) | set(recent_counts.index)
                    ref_aligned = [ref_counts.get(cat, 0) for cat in all_categories]
                    recent_aligned = [recent_counts.get(cat, 0) for cat in all_categories]
                    
                    if sum(ref_aligned) > 0 and sum(recent_aligned) > 0:
                        statistic, p_value = stats.chisquare(recent_aligned, ref_aligned)
                    else:
                        continue
                
                drift_scores.append(p_value)
                
                if p_value < drift_threshold:
                    affected_features.append(column)
                    
            except Exception as e:
                logger.warning(f"Error testing drift for column {column}: {e}")
                continue
        
        if not drift_scores:
            return None
        
        # Calculate overall drift score
        overall_drift_score = 1 - np.mean(drift_scores)  # Convert p-value to drift score
        is_drift_detected = len(affected_features) > 0
        
        return DriftResult(
            model_id=self.models[model_key]["model_id"],
            drift_type=DriftType.DATA_DRIFT,
            is_drift_detected=is_drift_detected,
            drift_score=overall_drift_score,
            confidence=0.95,
            affected_features=affected_features,
            timestamp=datetime.utcnow(),
            details={
                "individual_p_values": dict(zip(common_columns[:len(drift_scores)], drift_scores)),
                "drift_threshold": drift_threshold
            }
        )
    
    async def _detect_prediction_drift(self, model_key: str, recent_predictions: List[Dict[str, Any]]) -> Optional[DriftResult]:
        """Detect drift in prediction distributions."""
        if len(recent_predictions) < 100:
            return None
        
        # Split recent predictions into two periods
        mid_point = len(recent_predictions) // 2
        period1_preds = [p["prediction"] for p in recent_predictions[:mid_point]]
        period2_preds = [p["prediction"] for p in recent_predictions[mid_point:]]
        
        try:
            # Statistical test for prediction drift
            if all(isinstance(p, (int, float)) for p in period1_preds + period2_preds):
                # Numerical predictions - use KS test
                statistic, p_value = stats.ks_2samp(period1_preds, period2_preds)
            else:
                # Categorical predictions - use chi-square test
                from collections import Counter
                
                period1_counts = Counter(period1_preds)
                period2_counts = Counter(period2_preds)
                
                all_classes = set(period1_counts.keys()) | set(period2_counts.keys())
                period1_aligned = [period1_counts.get(cls, 0) for cls in all_classes]
                period2_aligned = [period2_counts.get(cls, 0) for cls in all_classes]
                
                statistic, p_value = stats.chisquare(period2_aligned, period1_aligned)
            
            drift_threshold = 0.05
            is_drift_detected = p_value < drift_threshold
            drift_score = 1 - p_value
            
            return DriftResult(
                model_id=self.models[model_key]["model_id"],
                drift_type=DriftType.PREDICTION_DRIFT,
                is_drift_detected=is_drift_detected,
                drift_score=drift_score,
                confidence=0.95,
                affected_features=["predictions"],
                timestamp=datetime.utcnow(),
                details={
                    "p_value": p_value,
                    "statistic": statistic,
                    "test_type": "ks_test" if isinstance(period1_preds[0], (int, float)) else "chi_square"
                }
            )
            
        except Exception as e:
            logger.warning(f"Error in prediction drift detection: {e}")
            return None
    
    async def _check_bias(self, model_key: str) -> None:
        """Check for bias in model predictions."""
        model_info = self.models[model_key]
        protected_attributes = model_info.get("protected_attributes", [])
        
        if not protected_attributes:
            return
        
        predictions = self.prediction_history[model_key]
        
        # Get recent predictions with ground truth
        recent_window = datetime.utcnow() - self.bias_check_interval
        recent_predictions = [
            p for p in predictions
            if p["timestamp"] >= recent_window and p["ground_truth"] is not None
        ]
        
        if len(recent_predictions) < 50:  # Need sufficient data for bias analysis
            return
        
        for protected_attr in protected_attributes:
            bias_result = await self._detect_bias(model_key, recent_predictions, protected_attr)
            if bias_result:
                self.bias_history[model_key].append(bias_result)
    
    async def _detect_bias(self, model_key: str, predictions: List[Dict[str, Any]], protected_attr: str) -> Optional[BiasResult]:
        """Detect bias for a specific protected attribute."""
        try:
            # Extract data for bias analysis
            data_for_bias = []
            for pred in predictions:
                input_data = pred["input_data"]
                if isinstance(input_data, dict) and protected_attr in input_data:
                    data_for_bias.append({
                        "protected_attr": input_data[protected_attr],
                        "prediction": pred["prediction"],
                        "ground_truth": pred["ground_truth"]
                    })
                elif isinstance(input_data, pd.DataFrame) and protected_attr in input_data.columns:
                    for _, row in input_data.iterrows():
                        data_for_bias.append({
                            "protected_attr": row[protected_attr],
                            "prediction": pred["prediction"],
                            "ground_truth": pred["ground_truth"]
                        })
            
            if len(data_for_bias) < 20:
                return None
            
            df = pd.DataFrame(data_for_bias)
            
            # Calculate fairness metrics
            fairness_metrics = {}
            
            # Demographic parity (statistical parity)
            groups = df.groupby("protected_attr")
            positive_rates = groups["prediction"].mean()
            
            if len(positive_rates) >= 2:
                max_rate = positive_rates.max()
                min_rate = positive_rates.min()
                demographic_parity = min_rate / max_rate if max_rate > 0 else 1.0
                fairness_metrics["demographic_parity"] = demographic_parity
            
            # Equalized odds (if binary classification)
            if set(df["ground_truth"].unique()).issubset({0, 1}) and set(df["prediction"].unique()).issubset({0, 1}):
                tpr_by_group = {}
                fpr_by_group = {}
                
                for group_name, group_data in groups:
                    tp = len(group_data[(group_data["ground_truth"] == 1) & (group_data["prediction"] == 1)])
                    fn = len(group_data[(group_data["ground_truth"] == 1) & (group_data["prediction"] == 0)])
                    fp = len(group_data[(group_data["ground_truth"] == 0) & (group_data["prediction"] == 1)])
                    tn = len(group_data[(group_data["ground_truth"] == 0) & (group_data["prediction"] == 0)])
                    
                    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                    
                    tpr_by_group[group_name] = tpr
                    fpr_by_group[group_name] = fpr
                
                if len(tpr_by_group) >= 2:
                    tpr_values = list(tpr_by_group.values())
                    fpr_values = list(fpr_by_group.values())
                    
                    tpr_difference = max(tpr_values) - min(tpr_values)
                    fpr_difference = max(fpr_values) - min(fpr_values)
                    
                    fairness_metrics["equalized_odds_tpr_diff"] = tpr_difference
                    fairness_metrics["equalized_odds_fpr_diff"] = fpr_difference
            
            # Calculate overall bias score
            bias_score = 0.0
            bias_detected = False
            
            if "demographic_parity" in fairness_metrics:
                bias_score = max(bias_score, 1 - fairness_metrics["demographic_parity"])
                bias_detected = bias_detected or fairness_metrics["demographic_parity"] < 0.8
            
            if "equalized_odds_tpr_diff" in fairness_metrics:
                bias_score = max(bias_score, fairness_metrics["equalized_odds_tpr_diff"])
                bias_detected = bias_detected or fairness_metrics["equalized_odds_tpr_diff"] > 0.2
            
            return BiasResult(
                model_id=self.models[model_key]["model_id"],
                protected_attribute=protected_attr,
                bias_detected=bias_detected,
                bias_score=bias_score,
                fairness_metrics=fairness_metrics,
                timestamp=datetime.utcnow(),
                details={
                    "sample_size": len(df),
                    "groups": df["protected_attr"].value_counts().to_dict()
                }
            )
            
        except Exception as e:
            logger.warning(f"Error in bias detection for {protected_attr}: {e}")
            return None
    
    async def _update_model_health(self, model_key: str) -> None:
        """Update overall model health assessment."""
        model_info = self.models[model_key]
        
        # Get recent metrics
        performance_metrics = self.performance_history[model_key][-10:] if self.performance_history[model_key] else []
        drift_results = self.drift_history[model_key][-5:] if self.drift_history[model_key] else []
        bias_results = self.bias_history[model_key][-5:] if self.bias_history[model_key] else []
        
        # Calculate health scores (0-1 scale)
        performance_health = self._calculate_performance_health(model_key, performance_metrics)
        drift_health = self._calculate_drift_health(drift_results)
        bias_health = self._calculate_bias_health(bias_results)
        operational_health = self._calculate_operational_health(model_key, performance_metrics)
        
        # Calculate overall health score
        weights = {"performance": 0.3, "drift": 0.3, "bias": 0.2, "operational": 0.2}
        overall_health = (
            weights["performance"] * performance_health +
            weights["drift"] * drift_health +
            weights["bias"] * bias_health +
            weights["operational"] * operational_health
        )
        
        # Determine status
        if overall_health >= 0.8:
            status = "healthy"
        elif overall_health >= 0.6:
            status = "degraded"
        else:
            status = "unhealthy"
        
        # Generate recommendations
        recommendations = self._generate_health_recommendations(
            performance_health, drift_health, bias_health, operational_health
        )
        
        # Create health assessment
        health = ModelHealth(
            model_id=model_info["model_id"],
            health_score=overall_health,
            status=status,
            performance_health=performance_health,
            drift_health=drift_health,
            bias_health=bias_health,
            operational_health=operational_health,
            recommendations=recommendations,
            timestamp=datetime.utcnow()
        )
        
        # Store health assessment
        health_file = self.storage_path / f"{model_key}_health.json"
        with open(health_file, 'w') as f:
            json.dump({
                "model_id": health.model_id,
                "health_score": health.health_score,
                "status": health.status,
                "performance_health": health.performance_health,
                "drift_health": health.drift_health,
                "bias_health": health.bias_health,
                "operational_health": health.operational_health,
                "recommendations": health.recommendations,
                "timestamp": health.timestamp.isoformat()
            }, f, indent=2)
        
        if status != "healthy":
            logger.warning(f"Model health alert for {model_key}: {status} (score: {overall_health:.2f})")
    
    def _calculate_performance_health(self, model_key: str, recent_metrics: List[ModelMetrics]) -> float:
        """Calculate performance health score."""
        if not recent_metrics:
            return 0.5  # Neutral score when no data
        
        latest_metrics = recent_metrics[-1]
        thresholds = self.performance_thresholds.get(model_key, {})
        model_type = self.models[model_key]["model_type"]
        
        scores = []
        
        # Check performance metrics based on model type
        if model_type == ModelType.CLASSIFICATION:
            accuracy = latest_metrics.performance_metrics.get("accuracy", 0)
            accuracy_threshold = thresholds.get("accuracy", 0.8)
            scores.append(min(accuracy / accuracy_threshold, 1.0))
            
            f1_score = latest_metrics.performance_metrics.get("f1_score", 0)
            f1_threshold = thresholds.get("f1_score", 0.7)
            scores.append(min(f1_score / f1_threshold, 1.0))
        
        elif model_type == ModelType.REGRESSION:
            r2_score = latest_metrics.performance_metrics.get("r2_score", 0)
            r2_threshold = thresholds.get("r2_score", 0.7)
            scores.append(min(r2_score / r2_threshold, 1.0))
        
        return np.mean(scores) if scores else 0.5
    
    def _calculate_drift_health(self, recent_drift_results: List[DriftResult]) -> float:
        """Calculate drift health score."""
        if not recent_drift_results:
            return 1.0  # No drift detected
        
        # Count recent drift detections
        recent_drifts = [r for r in recent_drift_results if r.is_drift_detected]
        
        if not recent_drifts:
            return 1.0
        
        # Penalize based on number and severity of drifts
        drift_penalty = min(len(recent_drifts) * 0.2, 0.8)
        avg_drift_score = np.mean([r.drift_score for r in recent_drifts])
        
        health_score = max(0.0, 1.0 - drift_penalty - (avg_drift_score * 0.3))
        return health_score
    
    def _calculate_bias_health(self, recent_bias_results: List[BiasResult]) -> float:
        """Calculate bias health score."""
        if not recent_bias_results:
            return 1.0  # No bias detected
        
        # Check for recent bias detections
        recent_bias = [r for r in recent_bias_results if r.bias_detected]
        
        if not recent_bias:
            return 1.0
        
        # Penalize based on bias severity
        avg_bias_score = np.mean([r.bias_score for r in recent_bias])
        health_score = max(0.0, 1.0 - avg_bias_score)
        
        return health_score
    
    def _calculate_operational_health(self, model_key: str, recent_metrics: List[ModelMetrics]) -> float:
        """Calculate operational health score."""
        if not recent_metrics:
            return 0.5
        
        latest_metrics = recent_metrics[-1]
        scores = []
        
        # Check error rate
        error_rate = latest_metrics.error_rate
        if error_rate <= 0.01:  # Less than 1% error rate
            scores.append(1.0)
        elif error_rate <= 0.05:  # Less than 5% error rate
            scores.append(0.7)
        else:
            scores.append(0.3)
        
        # Check latency
        latency_p95 = latest_metrics.latency_p95
        if latency_p95 <= 0.1:  # Less than 100ms
            scores.append(1.0)
        elif latency_p95 <= 0.5:  # Less than 500ms
            scores.append(0.8)
        elif latency_p95 <= 1.0:  # Less than 1 second
            scores.append(0.6)
        else:
            scores.append(0.3)
        
        return np.mean(scores)
    
    def _generate_health_recommendations(
        self,
        performance_health: float,
        drift_health: float,
        bias_health: float,
        operational_health: float
    ) -> List[str]:
        """Generate health improvement recommendations."""
        recommendations = []
        
        if performance_health < 0.7:
            recommendations.append("Model performance is below acceptable thresholds. Consider retraining with more recent data.")
        
        if drift_health < 0.7:
            recommendations.append("Data or prediction drift detected. Investigate input data changes and consider model updates.")
        
        if bias_health < 0.7:
            recommendations.append("Bias detected in model predictions. Review fairness constraints and consider bias mitigation techniques.")
        
        if operational_health < 0.7:
            recommendations.append("Operational issues detected. Check error rates, latency, and system resources.")
        
        if not recommendations:
            recommendations.append("Model is performing well across all health dimensions.")
        
        return recommendations
    
    def get_model_health(self, model_id: str, model_version: str) -> Optional[ModelHealth]:
        """Get current model health assessment."""
        model_key = f"{model_id}:{model_version}"
        health_file = self.storage_path / f"{model_key}_health.json"
        
        if not health_file.exists():
            return None
        
        try:
            with open(health_file, 'r') as f:
                health_data = json.load(f)
            
            return ModelHealth(
                model_id=health_data["model_id"],
                health_score=health_data["health_score"],
                status=health_data["status"],
                performance_health=health_data["performance_health"],
                drift_health=health_data["drift_health"],
                bias_health=health_data["bias_health"],
                operational_health=health_data["operational_health"],
                recommendations=health_data["recommendations"],
                timestamp=datetime.fromisoformat(health_data["timestamp"])
            )
            
        except Exception as e:
            logger.error(f"Error loading model health: {e}")
            return None
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        total_predictions = sum(len(predictions) for predictions in self.prediction_history.values())
        total_drift_detections = sum(
            len([r for r in drift_results if r.is_drift_detected])
            for drift_results in self.drift_history.values()
        )
        total_bias_detections = sum(
            len([r for r in bias_results if r.bias_detected])
            for bias_results in self.bias_history.values()
        )
        
        return {
            "registered_models": len(self.models),
            "total_predictions": total_predictions,
            "drift_detections": total_drift_detections,
            "bias_detections": total_bias_detections,
            "monitoring_tasks": len(self.monitoring_tasks),
            "explainability_enabled": self.enable_explainability,
            "explainers_active": len(self.explainers)
        }