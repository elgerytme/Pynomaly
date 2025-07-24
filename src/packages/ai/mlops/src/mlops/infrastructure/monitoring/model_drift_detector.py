"""
Model Drift Detection System

Comprehensive system for detecting data drift, model performance degradation,
and distributional changes in production ML models.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import structlog

from mlops.domain.entities.model import Model, ModelVersion
from mlops.infrastructure.monitoring.pipeline_monitor import PipelineAlert, AlertSeverity


class DriftType(Enum):
    """Types of drift that can be detected."""
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    PREDICTION_DRIFT = "prediction_drift"
    PERFORMANCE_DRIFT = "performance_drift"


class DriftSeverity(Enum):
    """Severity levels for drift detection."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DriftDetectionResult:
    """Result of drift detection analysis."""
    model_id: str
    drift_type: DriftType
    severity: DriftSeverity
    drift_score: float
    threshold: float
    detected_at: datetime = field(default_factory=datetime.utcnow)
    features_affected: List[str] = field(default_factory=list)
    statistical_test: str = ""
    p_value: Optional[float] = None
    confidence: float = 0.0
    description: str = ""
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MonitoringWindow:
    """Data window for monitoring."""
    reference_data: pd.DataFrame
    current_data: pd.DataFrame
    window_start: datetime
    window_end: datetime
    sample_size: int


class StatisticalTest:
    """Statistical tests for drift detection."""
    
    @staticmethod
    def kolmogorov_smirnov_test(reference_data: np.ndarray, current_data: np.ndarray) -> Tuple[float, float]:
        """Perform Kolmogorov-Smirnov test for distribution comparison."""
        try:
            statistic, p_value = stats.ks_2samp(reference_data, current_data)
            return statistic, p_value
        except Exception as e:
            warnings.warn(f"KS test failed: {e}")
            return 0.0, 1.0
    
    @staticmethod
    def chi_square_test(reference_data: np.ndarray, current_data: np.ndarray, bins: int = 10) -> Tuple[float, float]:
        """Perform Chi-square test for categorical data."""
        try:
            # Create histograms
            ref_hist, bin_edges = np.histogram(reference_data, bins=bins)
            curr_hist, _ = np.histogram(current_data, bins=bin_edges)
            
            # Add small epsilon to avoid zero values
            ref_hist = ref_hist + 1e-10
            curr_hist = curr_hist + 1e-10
            
            # Normalize
            ref_hist = ref_hist / np.sum(ref_hist)
            curr_hist = curr_hist / np.sum(curr_hist)
            
            # Chi-square test
            statistic, p_value = stats.chisquare(curr_hist, ref_hist)
            return statistic, p_value
        except Exception as e:
            warnings.warn(f"Chi-square test failed: {e}")
            return 0.0, 1.0
    
    @staticmethod
    def earth_movers_distance(reference_data: np.ndarray, current_data: np.ndarray) -> float:
        """Calculate Earth Mover's Distance (Wasserstein distance)."""
        try:
            return stats.wasserstein_distance(reference_data, current_data)
        except Exception as e:
            warnings.warn(f"EMD calculation failed: {e}")
            return 0.0
    
    @staticmethod
    def jensen_shannon_divergence(reference_data: np.ndarray, current_data: np.ndarray, bins: int = 10) -> float:
        """Calculate Jensen-Shannon divergence."""
        try:
            # Create probability distributions
            ref_hist, bin_edges = np.histogram(reference_data, bins=bins, density=True)
            curr_hist, _ = np.histogram(current_data, bins=bin_edges, density=True)
            
            # Normalize to probabilities
            ref_prob = ref_hist / np.sum(ref_hist)
            curr_prob = curr_hist / np.sum(curr_hist)
            
            # Add small epsilon to avoid log(0)
            ref_prob = ref_prob + 1e-10
            curr_prob = curr_prob + 1e-10
            
            # Calculate Jensen-Shannon divergence
            m = 0.5 * (ref_prob + curr_prob)
            js_div = 0.5 * stats.entropy(ref_prob, m) + 0.5 * stats.entropy(curr_prob, m)
            
            return js_div
        except Exception as e:
            warnings.warn(f"JS divergence calculation failed: {e}")
            return 0.0


class ModelDriftDetector:
    """Comprehensive model drift detection system."""
    
    def __init__(self, 
                 reference_window_days: int = 30,
                 monitoring_window_hours: int = 24,
                 drift_threshold: float = 0.05,
                 performance_threshold: float = 0.1):
        self.reference_window_days = reference_window_days
        self.monitoring_window_hours = monitoring_window_hours
        self.drift_threshold = drift_threshold
        self.performance_threshold = performance_threshold
        
        self.logger = structlog.get_logger(__name__)
        
        # Detection methods
        self.statistical_tests = StatisticalTest()
        
        # Model monitoring state
        self.model_baselines: Dict[str, Dict] = {}
        self.detection_history: Dict[str, List[DriftDetectionResult]] = {}
        self.monitoring_tasks: Dict[str, asyncio.Task] = {}
        
        # Configuration
        self.detection_methods = {
            DriftType.DATA_DRIFT: self._detect_data_drift,
            DriftType.CONCEPT_DRIFT: self._detect_concept_drift,
            DriftType.PREDICTION_DRIFT: self._detect_prediction_drift,
            DriftType.PERFORMANCE_DRIFT: self._detect_performance_drift
        }
    
    async def register_model_for_monitoring(self, 
                                          model_id: str, 
                                          baseline_data: pd.DataFrame,
                                          baseline_predictions: np.ndarray = None,
                                          baseline_labels: np.ndarray = None,
                                          feature_columns: List[str] = None) -> None:
        """Register a model for drift monitoring."""
        
        # Store baseline information
        self.model_baselines[model_id] = {
            'baseline_data': baseline_data.copy(),
            'baseline_predictions': baseline_predictions.copy() if baseline_predictions is not None else None,
            'baseline_labels': baseline_labels.copy() if baseline_labels is not None else None,
            'feature_columns': feature_columns or list(baseline_data.columns),
            'registered_at': datetime.utcnow(),
            'data_statistics': self._calculate_data_statistics(baseline_data),
            'prediction_statistics': self._calculate_prediction_statistics(baseline_predictions) if baseline_predictions is not None else None
        }
        
        # Initialize detection history
        self.detection_history[model_id] = []
        
        self.logger.info(
            "Model registered for drift monitoring",
            model_id=model_id,
            baseline_samples=len(baseline_data),
            feature_count=len(feature_columns or baseline_data.columns)
        )
    
    async def start_continuous_monitoring(self, model_id: str, monitoring_interval_minutes: int = 60) -> None:
        """Start continuous drift monitoring for a model."""
        if model_id not in self.model_baselines:
            raise ValueError(f"Model {model_id} not registered for monitoring")
        
        if model_id in self.monitoring_tasks:
            # Stop existing monitoring
            await self.stop_continuous_monitoring(model_id)
        
        # Start monitoring task
        task = asyncio.create_task(
            self._continuous_monitoring_loop(model_id, monitoring_interval_minutes)
        )
        self.monitoring_tasks[model_id] = task
        
        self.logger.info(
            "Started continuous drift monitoring",
            model_id=model_id,
            interval_minutes=monitoring_interval_minutes
        )
    
    async def stop_continuous_monitoring(self, model_id: str) -> None:
        """Stop continuous drift monitoring for a model."""
        if model_id in self.monitoring_tasks:
            task = self.monitoring_tasks[model_id]
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            del self.monitoring_tasks[model_id]
            
            self.logger.info(
                "Stopped continuous drift monitoring",
                model_id=model_id
            )
    
    async def _continuous_monitoring_loop(self, model_id: str, interval_minutes: int) -> None:
        """Continuous monitoring loop for a model."""
        while True:
            try:
                # This would typically fetch recent data from a data store
                # For now, we'll simulate the monitoring
                await asyncio.sleep(interval_minutes * 60)
                
                # In a real implementation, you would:
                # 1. Fetch recent production data
                # 2. Run drift detection
                # 3. Generate alerts if drift is detected
                
                self.logger.debug(
                    "Continuous monitoring check completed",
                    model_id=model_id
                )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(
                    "Error in continuous monitoring loop",
                    model_id=model_id,
                    error=str(e)
                )
                await asyncio.sleep(300)  # Wait 5 minutes before retry
    
    async def detect_drift(self, 
                          model_id: str, 
                          current_data: pd.DataFrame,
                          current_predictions: np.ndarray = None,
                          current_labels: np.ndarray = None,
                          drift_types: List[DriftType] = None) -> List[DriftDetectionResult]:
        """Detect drift for a model using current data."""
        
        if model_id not in self.model_baselines:
            raise ValueError(f"Model {model_id} not registered for monitoring")
        
        baseline = self.model_baselines[model_id]
        drift_types = drift_types or list(DriftType)
        
        results = []
        
        for drift_type in drift_types:
            if drift_type in self.detection_methods:
                try:
                    result = await self.detection_methods[drift_type](
                        model_id, baseline, current_data, current_predictions, current_labels
                    )
                    if result:
                        results.append(result)
                        
                        # Store in history
                        self.detection_history[model_id].append(result)
                        
                        # Log detection
                        self.logger.info(
                            "Drift detected",
                            model_id=model_id,
                            drift_type=drift_type.value,
                            severity=result.severity.value,
                            drift_score=result.drift_score,
                            threshold=result.threshold
                        )
                        
                except Exception as e:
                    self.logger.error(
                        "Error detecting drift",
                        model_id=model_id,
                        drift_type=drift_type.value,
                        error=str(e)
                    )
        
        return results
    
    async def _detect_data_drift(self, 
                               model_id: str, 
                               baseline: Dict, 
                               current_data: pd.DataFrame,
                               current_predictions: np.ndarray = None,
                               current_labels: np.ndarray = None) -> Optional[DriftDetectionResult]:
        """Detect data drift using statistical tests."""
        
        baseline_data = baseline['baseline_data']
        feature_columns = baseline['feature_columns']
        
        drift_scores = []
        affected_features = []
        test_results = []
        
        for feature in feature_columns:
            if feature not in current_data.columns:
                continue
            
            # Get feature data
            baseline_feature = baseline_data[feature].dropna()
            current_feature = current_data[feature].dropna()
            
            if len(baseline_feature) == 0 or len(current_feature) == 0:
                continue
            
            # Choose appropriate test based on data type
            if pd.api.types.is_numeric_dtype(baseline_feature):
                # Numerical feature - use KS test
                ks_stat, p_value = self.statistical_tests.kolmogorov_smirnov_test(
                    baseline_feature.values, current_feature.values
                )
                
                drift_score = ks_stat
                test_name = "Kolmogorov-Smirnov"
                
            else:
                # Categorical feature - use Chi-square test
                try:
                    # Convert to numeric for testing
                    baseline_numeric = pd.Categorical(baseline_feature).codes
                    current_numeric = pd.Categorical(current_feature).codes
                    
                    chi2_stat, p_value = self.statistical_tests.chi_square_test(
                        baseline_numeric, current_numeric
                    )
                    
                    drift_score = chi2_stat
                    test_name = "Chi-square"
                    
                except Exception:
                    continue
            
            drift_scores.append(drift_score)
            test_results.append((feature, test_name, drift_score, p_value))
            
            # Check if feature shows drift
            if p_value < self.drift_threshold:
                affected_features.append(feature)
        
        # Calculate overall drift score
        if not drift_scores:
            return None
        
        overall_drift_score = np.mean(drift_scores)
        min_p_value = min([result[3] for result in test_results])
        
        # Determine severity
        if min_p_value < 0.001:
            severity = DriftSeverity.CRITICAL
        elif min_p_value < 0.01:
            severity = DriftSeverity.HIGH
        elif min_p_value < 0.05:
            severity = DriftSeverity.MEDIUM
        else:
            severity = DriftSeverity.LOW
        
        # Generate recommendations
        recommendations = []
        if affected_features:
            recommendations.extend([
                f"Investigate changes in features: {', '.join(affected_features[:5])}",
                "Consider retraining the model with recent data",
                "Review data collection and preprocessing pipelines"
            ])
        
        # Only return result if significant drift detected
        if min_p_value < self.drift_threshold:
            return DriftDetectionResult(
                model_id=model_id,
                drift_type=DriftType.DATA_DRIFT,
                severity=severity,
                drift_score=overall_drift_score,
                threshold=self.drift_threshold,
                features_affected=affected_features,
                statistical_test="Multiple tests",
                p_value=min_p_value,
                confidence=1.0 - min_p_value,
                description=f"Data drift detected in {len(affected_features)} features",
                recommendations=recommendations,
                metadata={
                    "test_results": test_results,
                    "total_features_tested": len(drift_scores),
                    "baseline_samples": len(baseline_data),
                    "current_samples": len(current_data)
                }
            )
        
        return None
    
    async def _detect_concept_drift(self, 
                                  model_id: str, 
                                  baseline: Dict, 
                                  current_data: pd.DataFrame,
                                  current_predictions: np.ndarray = None,
                                  current_labels: np.ndarray = None) -> Optional[DriftDetectionResult]:
        """Detect concept drift (relationship between features and target)."""
        
        # Concept drift requires both current predictions and labels
        if current_predictions is None or current_labels is None:
            return None
        
        baseline_predictions = baseline.get('baseline_predictions')
        baseline_labels = baseline.get('baseline_labels')
        
        if baseline_predictions is None or baseline_labels is None:
            return None
        
        # Calculate performance metrics for baseline and current data
        baseline_accuracy = accuracy_score(baseline_labels, baseline_predictions)
        current_accuracy = accuracy_score(current_labels, current_predictions)
        
        # Calculate performance degradation
        performance_drop = baseline_accuracy - current_accuracy
        drift_score = abs(performance_drop)
        
        # Determine severity based on performance drop
        if performance_drop > 0.2:
            severity = DriftSeverity.CRITICAL
        elif performance_drop > 0.1:
            severity = DriftSeverity.HIGH
        elif performance_drop > 0.05:
            severity = DriftSeverity.MEDIUM
        else:
            severity = DriftSeverity.LOW
        
        # Generate recommendations
        recommendations = []
        if performance_drop > self.performance_threshold:
            recommendations.extend([
                "Model performance has significantly degraded",
                "Consider immediate model retraining",
                "Investigate changes in target variable distribution",
                "Review feature engineering pipeline"
            ])
        
        # Only return result if significant concept drift detected
        if performance_drop > self.performance_threshold:
            return DriftDetectionResult(
                model_id=model_id,
                drift_type=DriftType.CONCEPT_DRIFT,
                severity=severity,
                drift_score=drift_score,
                threshold=self.performance_threshold,
                statistical_test="Performance comparison",
                confidence=drift_score,
                description=f"Model accuracy dropped from {baseline_accuracy:.3f} to {current_accuracy:.3f}",
                recommendations=recommendations,
                metadata={
                    "baseline_accuracy": baseline_accuracy,
                    "current_accuracy": current_accuracy,
                    "performance_drop": performance_drop,
                    "baseline_samples": len(baseline_labels),
                    "current_samples": len(current_labels)
                }
            )
        
        return None
    
    async def _detect_prediction_drift(self, 
                                     model_id: str, 
                                     baseline: Dict, 
                                     current_data: pd.DataFrame,
                                     current_predictions: np.ndarray = None,
                                     current_labels: np.ndarray = None) -> Optional[DriftDetectionResult]:
        """Detect drift in model predictions."""
        
        if current_predictions is None:
            return None
        
        baseline_predictions = baseline.get('baseline_predictions')
        if baseline_predictions is None:
            return None
        
        # Compare prediction distributions
        ks_stat, p_value = self.statistical_tests.kolmogorov_smirnov_test(
            baseline_predictions, current_predictions
        )
        
        # Calculate Jensen-Shannon divergence for additional insight
        js_divergence = self.statistical_tests.jensen_shannon_divergence(
            baseline_predictions, current_predictions
        )
        
        drift_score = max(ks_stat, js_divergence)
        
        # Determine severity
        if p_value < 0.001 or js_divergence > 0.5:
            severity = DriftSeverity.CRITICAL
        elif p_value < 0.01 or js_divergence > 0.3:
            severity = DriftSeverity.HIGH
        elif p_value < 0.05 or js_divergence > 0.1:
            severity = DriftSeverity.MEDIUM
        else:
            severity = DriftSeverity.LOW
        
        # Generate recommendations
        recommendations = []
        if p_value < self.drift_threshold:
            recommendations.extend([
                "Model prediction distribution has shifted significantly",
                "Investigate input data changes",
                "Consider model recalibration",
                "Review model serving infrastructure"
            ])
        
        # Only return result if significant prediction drift detected
        if p_value < self.drift_threshold:
            return DriftDetectionResult(
                model_id=model_id,
                drift_type=DriftType.PREDICTION_DRIFT,
                severity=severity,
                drift_score=drift_score,
                threshold=self.drift_threshold,
                statistical_test="KS test + JS divergence",
                p_value=p_value,
                confidence=1.0 - p_value,
                description=f"Prediction distribution drift detected (KS={ks_stat:.3f}, JS={js_divergence:.3f})",
                recommendations=recommendations,
                metadata={
                    "ks_statistic": ks_stat,
                    "js_divergence": js_divergence,
                    "baseline_pred_mean": np.mean(baseline_predictions),
                    "current_pred_mean": np.mean(current_predictions),
                    "baseline_pred_std": np.std(baseline_predictions),
                    "current_pred_std": np.std(current_predictions)
                }
            )
        
        return None
    
    async def _detect_performance_drift(self, 
                                      model_id: str, 
                                      baseline: Dict, 
                                      current_data: pd.DataFrame,
                                      current_predictions: np.ndarray = None,
                                      current_labels: np.ndarray = None) -> Optional[DriftDetectionResult]:
        """Detect performance drift using multiple metrics."""
        
        if current_predictions is None or current_labels is None:
            return None
        
        baseline_predictions = baseline.get('baseline_predictions')
        baseline_labels = baseline.get('baseline_labels')
        
        if baseline_predictions is None or baseline_labels is None:
            return None
        
        # Calculate multiple performance metrics
        metrics = {}
        
        # Accuracy
        baseline_acc = accuracy_score(baseline_labels, baseline_predictions)
        current_acc = accuracy_score(current_labels, current_predictions)
        metrics['accuracy'] = {'baseline': baseline_acc, 'current': current_acc, 'drop': baseline_acc - current_acc}
        
        # Precision (handle binary/multiclass)
        try:
            baseline_prec = precision_score(baseline_labels, baseline_predictions, average='weighted')
            current_prec = precision_score(current_labels, current_predictions, average='weighted')
            metrics['precision'] = {'baseline': baseline_prec, 'current': current_prec, 'drop': baseline_prec - current_prec}
        except:
            pass
        
        # Recall
        try:
            baseline_rec = recall_score(baseline_labels, baseline_predictions, average='weighted')
            current_rec = recall_score(current_labels, current_predictions, average='weighted')
            metrics['recall'] = {'baseline': baseline_rec, 'current': current_rec, 'drop': baseline_rec - current_rec}
        except:
            pass
        
        # F1 Score
        try:
            baseline_f1 = f1_score(baseline_labels, baseline_predictions, average='weighted')
            current_f1 = f1_score(current_labels, current_predictions, average='weighted')
            metrics['f1'] = {'baseline': baseline_f1, 'current': current_f1, 'drop': baseline_f1 - current_f1}
        except:
            pass
        
        # Calculate overall performance drop
        performance_drops = [metric['drop'] for metric in metrics.values() if metric['drop'] > 0]
        
        if not performance_drops:
            return None
        
        max_drop = max(performance_drops)
        avg_drop = np.mean(performance_drops)
        drift_score = max_drop
        
        # Determine severity
        if max_drop > 0.2:
            severity = DriftSeverity.CRITICAL
        elif max_drop > 0.1:
            severity = DriftSeverity.HIGH
        elif max_drop > 0.05:
            severity = DriftSeverity.MEDIUM
        else:
            severity = DriftSeverity.LOW
        
        # Generate recommendations
        recommendations = []
        if max_drop > self.performance_threshold:
            recommendations.extend([
                f"Model performance degraded significantly (max drop: {max_drop:.3f})",
                "Immediate model retraining recommended",
                "Investigate data quality issues",
                "Consider ensemble or fallback models"
            ])
        
        # Only return result if significant performance drift detected
        if max_drop > self.performance_threshold:
            return DriftDetectionResult(
                model_id=model_id,
                drift_type=DriftType.PERFORMANCE_DRIFT,
                severity=severity,
                drift_score=drift_score,
                threshold=self.performance_threshold,
                statistical_test="Performance metrics comparison",
                confidence=drift_score,
                description=f"Performance degradation detected (max drop: {max_drop:.3f})",
                recommendations=recommendations,
                metadata={
                    "metrics": metrics,
                    "max_performance_drop": max_drop,
                    "avg_performance_drop": avg_drop,
                    "baseline_samples": len(baseline_labels),
                    "current_samples": len(current_labels)
                }
            )
        
        return None
    
    def _calculate_data_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate statistical summary of data."""
        stats = {}
        
        for column in data.columns:
            if pd.api.types.is_numeric_dtype(data[column]):
                stats[column] = {
                    'type': 'numeric',
                    'mean': float(data[column].mean()),
                    'std': float(data[column].std()),
                    'min': float(data[column].min()),
                    'max': float(data[column].max()),
                    'median': float(data[column].median()),
                    'percentiles': {
                        '25': float(data[column].quantile(0.25)),
                        '75': float(data[column].quantile(0.75))
                    }
                }
            else:
                value_counts = data[column].value_counts()
                stats[column] = {
                    'type': 'categorical',
                    'unique_values': int(data[column].nunique()),
                    'most_frequent': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                    'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                    'categories': value_counts.head(10).to_dict()
                }
        
        return stats
    
    def _calculate_prediction_statistics(self, predictions: np.ndarray) -> Dict[str, Any]:
        """Calculate statistical summary of predictions."""
        return {
            'mean': float(np.mean(predictions)),
            'std': float(np.std(predictions)),
            'min': float(np.min(predictions)),
            'max': float(np.max(predictions)),
            'median': float(np.median(predictions)),
            'percentiles': {
                '25': float(np.percentile(predictions, 25)),
                '75': float(np.percentile(predictions, 75))
            }
        }
    
    async def get_drift_history(self, model_id: str, days: int = 30) -> List[Dict[str, Any]]:
        """Get drift detection history for a model."""
        if model_id not in self.detection_history:
            return []
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        history = [
            {
                'detected_at': result.detected_at.isoformat(),
                'drift_type': result.drift_type.value,
                'severity': result.severity.value,
                'drift_score': result.drift_score,
                'threshold': result.threshold,
                'features_affected': result.features_affected,
                'statistical_test': result.statistical_test,
                'p_value': result.p_value,
                'confidence': result.confidence,
                'description': result.description,
                'recommendations': result.recommendations
            }
            for result in self.detection_history[model_id]
            if result.detected_at >= cutoff_date
        ]
        
        return history
    
    async def get_monitoring_status(self) -> Dict[str, Any]:
        """Get overall drift monitoring status."""
        return {
            'monitored_models': len(self.model_baselines),
            'active_monitoring_tasks': len(self.monitoring_tasks),
            'total_drift_detections': sum(len(history) for history in self.detection_history.values()),
            'model_summaries': [
                {
                    'model_id': model_id,
                    'registered_at': baseline['registered_at'].isoformat(),
                    'feature_count': len(baseline['feature_columns']),
                    'baseline_samples': len(baseline['baseline_data']),
                    'drift_detections': len(self.detection_history.get(model_id, [])),
                    'is_actively_monitored': model_id in self.monitoring_tasks
                }
                for model_id, baseline in self.model_baselines.items()
            ]
        }