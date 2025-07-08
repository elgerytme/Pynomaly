"""Model Performance Degradation Detector Service.

This service provides multiple algorithms for detecting performance degradation
in machine learning models by comparing current metrics against baseline metrics.
"""

from __future__ import annotations

from typing import Dict, Any, Union, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from scipy import stats
from datetime import datetime
import logging

from pynomaly.domain.entities.model_performance import ModelPerformanceMetrics, ModelPerformanceBaseline
from pynomaly.domain.entities.alert import AlertSeverity, AlertSource, AlertType, NotificationChannel

logger = logging.getLogger(__name__)

class DetectionAlgorithm(Enum):
    """Enumeration of supported degradation detection algorithms."""
    SIMPLE_THRESHOLD = "simple_threshold"
    STATISTICAL = "statistical"
    ML_BASED = "ml_based"


@dataclass
class DegradationDetectorConfig:
    """Configuration for the degradation detector.
    
    Attributes:
        algorithm: The detection algorithm to use
        delta: Threshold delta for simple threshold algorithm (default: 0.1)
        confidence: Confidence level for statistical tests (default: 0.95)
        statistical_method: Statistical method to use ('z_score' or 't_test')
        ml_model: Optional ML model for ML-based detection
        metric_weights: Optional weights for different metrics
    """
    algorithm: DetectionAlgorithm
    delta: float = 0.1
    confidence: float = 0.95
    statistical_method: str = "z_score"
    ml_model: Optional[Any] = None
    metric_weights: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not isinstance(self.algorithm, DetectionAlgorithm):
            if isinstance(self.algorithm, str):
                self.algorithm = DetectionAlgorithm(self.algorithm)
            else:
                raise ValueError(f"Invalid algorithm type: {type(self.algorithm)}")
        
        if not (0.0 < self.delta < 1.0):
            raise ValueError(f"Delta must be between 0.0 and 1.0, got {self.delta}")
        
        if not (0.0 < self.confidence < 1.0):
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")
        
        if self.statistical_method not in ["z_score", "t_test"]:
            raise ValueError(f"Statistical method must be 'z_score' or 't_test', got {self.statistical_method}")
        
        if self.metric_weights:
            total_weight = sum(self.metric_weights.values())
            if not np.isclose(total_weight, 1.0):
                raise ValueError(f"Metric weights must sum to 1.0, got {total_weight}")


@dataclass
class DegradationDetails:
    """Details about performance degradation for a specific metric.
    
    Attributes:
        metric_name: Name of the affected metric
        current_value: Current value of the metric
        baseline_value: Baseline value for comparison
        deviation: Absolute deviation from baseline
        relative_deviation: Relative deviation as percentage
        statistical_significance: Statistical significance (p-value, z-score, etc.)
    """
    metric_name: str
    current_value: float
    baseline_value: float
    deviation: float
    relative_deviation: float
    statistical_significance: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        """Calculate derived values after initialization."""
        self.deviation = self.baseline_value - self.current_value
        if self.baseline_value != 0:
            self.relative_deviation = (self.deviation / self.baseline_value) * 100
        else:
            self.relative_deviation = 0.0


@dataclass
class DegradationResult:
    """Result of degradation detection.
    
    Attributes:
        degrade_flag: Whether degradation was detected
        affected_metrics: List of affected metrics with details
        detection_algorithm: Algorithm used for detection
        overall_severity: Overall severity score (0-1)
        metadata: Additional metadata about the detection
    """
    degrade_flag: bool
    affected_metrics: List[DegradationDetails]
    detection_algorithm: DetectionAlgorithm
    overall_severity: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate overall severity after initialization."""
        if self.affected_metrics:
            # Calculate severity as average relative deviation
            self.overall_severity = np.mean([
                detail.relative_deviation for detail in self.affected_metrics
            ]) / 100.0  # Normalize to 0-1 scale
        else:
            self.overall_severity = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            "degrade_flag": self.degrade_flag,
            "affected_metrics": [
                {
                    "metric_name": detail.metric_name,
                    "current_value": detail.current_value,
                    "baseline_value": detail.baseline_value,
                    "deviation": detail.deviation,
                    "relative_deviation": detail.relative_deviation,
                    "statistical_significance": detail.statistical_significance
                }
                for detail in self.affected_metrics
            ],
            "detection_algorithm": self.detection_algorithm.value,
            "overall_severity": self.overall_severity,
            "metadata": self.metadata
        }

class ModelPerformanceDegradationDetector:
    """Service for detecting model performance degradation.
    
    This service provides multiple algorithms for detecting performance degradation
    in machine learning models by comparing current metrics against baseline metrics.
    """
    
    def __init__(self, config: DegradationDetectorConfig):
        """Initialize the degradation detector.
        
        Args:
            config: Configuration object containing algorithm settings
        """
        self.config = config
        self._metric_names = ['accuracy', 'precision', 'recall', 'f1']
    
    async def detect(self,
                      current_metrics: ModelPerformanceMetrics,
                      baseline: ModelPerformanceBaseline,
                      model_id: str,
                      model_name: str,
                      notification_channels: Optional[List[NotificationChannel]] = None) -> DegradationResult:
        """Detect performance degradation using the configured algorithm.
        
        Args:
            current_metrics: Current performance metrics
            baseline: Baseline performance metrics for comparison
            
        Returns:
            DegradationResult containing detection results and details
        """
        degradation_result = self._run_detection_algorithm(current_metrics, baseline)

        if degradation_result.degrade_flag:
            # Log detected degradation
            logger.info(f"Performance degradation detected for model {model_name}")

            # Instantiate PerformanceAlertService (simplified version)
            from pynomaly.application.services.performance_alert_service import PerformanceAlertService
            from pynomaly.application.services.intelligent_alert_service import IntelligentAlertService
            
            # Create a simplified alert service
            intelligent_service = IntelligentAlertService()
            performance_alert_service = PerformanceAlertService(intelligent_service)

            try:
                # Create a performance alert
                alert = await performance_alert_service.create_performance_alert(
                    degradation_result=degradation_result,
                    model_id=model_id,
                    model_name=model_name,
                    notification_channels=notification_channels
                )
                logger.info(f"✓ Alert created: {alert.name} (ID: {alert.id})")
                logger.info(f"  Alert severity: {alert.severity.value}")
                logger.info(f"  Alert source: {alert.source}")
                logger.info(f"  Notifications via: {[ch.value for ch in (notification_channels or [])]}")
            except Exception as e:
                logger.error(f"Failed to create alert: {e}")
                # Continue with detection even if alerting fails

        return degradation_result

    def _run_detection_algorithm(self,
                                current_metrics: ModelPerformanceMetrics,
                                baseline: ModelPerformanceBaseline) -> DegradationResult:
        if self.config.algorithm == DetectionAlgorithm.SIMPLE_THRESHOLD:
            return self._simple_threshold_detection(current_metrics, baseline)
        elif self.config.algorithm == DetectionAlgorithm.STATISTICAL:
            return self._statistical_detection(current_metrics, baseline)
        elif self.config.algorithm == DetectionAlgorithm.ML_BASED:
            return self._ml_based_detection(current_metrics, baseline)
        else:
            raise ValueError(f"Unsupported algorithm: {self.config.algorithm}")
    
    def _simple_threshold_detection(self, 
                                  current: ModelPerformanceMetrics, 
                                  baseline: ModelPerformanceBaseline) -> DegradationResult:
        """Detect degradation using simple threshold algorithm.
        
        Args:
            current: Current performance metrics
            baseline: Baseline performance metrics
            
        Returns:
            DegradationResult with detection results
        """
        affected_metrics = []
        threshold = baseline.mean * (1 - self.config.delta)
        
        for metric_name in self._metric_names:
            current_value = getattr(current, metric_name)
            if current_value < threshold:
                detail = DegradationDetails(
                    metric_name=metric_name,
                    current_value=current_value,
                    baseline_value=baseline.mean,
                    deviation=0.0,  # Will be calculated in __post_init__
                    relative_deviation=0.0  # Will be calculated in __post_init__
                )
                affected_metrics.append(detail)
        
        return DegradationResult(
            degrade_flag=len(affected_metrics) > 0,
            affected_metrics=affected_metrics,
            detection_algorithm=self.config.algorithm,
            metadata={
                "threshold": threshold,
                "delta": self.config.delta
            }
        )
    
    def _statistical_detection(self, 
                             current: ModelPerformanceMetrics, 
                             baseline: ModelPerformanceBaseline) -> DegradationResult:
        """Detect degradation using statistical methods (z-score or t-test).
        
        Args:
            current: Current performance metrics
            baseline: Baseline performance metrics
            
        Returns:
            DegradationResult with detection results
        """
        affected_metrics = []
        alpha = 1 - self.config.confidence
        
        for metric_name in self._metric_names:
            current_value = getattr(current, metric_name)
            
            if self.config.statistical_method == "z_score":
                if baseline.std == 0:
                    # If std is 0, use simple threshold as fallback
                    threshold = baseline.mean * (1 - self.config.delta)
                    is_degraded = current_value < threshold
                    significance = {"method": "threshold_fallback", "threshold": threshold}
                else:
                    z_score = (current_value - baseline.mean) / baseline.std
                    p_value = stats.norm.sf(abs(z_score)) * 2  # Two-tailed test
                    is_degraded = p_value < alpha and z_score < 0  # Degradation means lower performance
                    significance = {"z_score": z_score, "p_value": p_value, "alpha": alpha}
            
            elif self.config.statistical_method == "t_test":
                # For t-test, we need degrees of freedom (assuming sample size)
                # This is a simplified implementation
                if baseline.std == 0:
                    threshold = baseline.mean * (1 - self.config.delta)
                    is_degraded = current_value < threshold
                    significance = {"method": "threshold_fallback", "threshold": threshold}
                else:
                    # Assuming single sample t-test
                    t_stat = (current_value - baseline.mean) / baseline.std
                    df = 29  # Assuming 30 samples (df = n-1)
                    p_value = stats.t.sf(abs(t_stat), df) * 2  # Two-tailed test
                    is_degraded = p_value < alpha and t_stat < 0
                    significance = {"t_stat": t_stat, "p_value": p_value, "df": df, "alpha": alpha}
            
            if is_degraded:
                detail = DegradationDetails(
                    metric_name=metric_name,
                    current_value=current_value,
                    baseline_value=baseline.mean,
                    deviation=0.0,  # Will be calculated in __post_init__
                    relative_deviation=0.0,  # Will be calculated in __post_init__
                    statistical_significance=significance
                )
                affected_metrics.append(detail)
        
        return DegradationResult(
            degrade_flag=len(affected_metrics) > 0,
            affected_metrics=affected_metrics,
            detection_algorithm=self.config.algorithm,
            metadata={
                "statistical_method": self.config.statistical_method,
                "confidence": self.config.confidence,
                "alpha": alpha
            }
        )
    
    def _ml_based_detection(self, 
                          current: ModelPerformanceMetrics, 
                          baseline: ModelPerformanceBaseline) -> DegradationResult:
        """Detect degradation using ML-based methods (placeholder).
        
        Args:
            current: Current performance metrics
            baseline: Baseline performance metrics
            
        Returns:
            DegradationResult with detection results
        """
        # This is a placeholder implementation for ML-based detection
        # In a real implementation, this would use a trained classifier/regressor
        # to predict degradation based on historical patterns
        
        affected_metrics = []
        
        if self.config.ml_model is not None:
            # Example implementation using a hypothetical ML model
            # The model would be trained on historical data with labels
            # indicating normal vs degraded performance
            
            # Prepare features from current metrics
            features = np.array([
                current.accuracy,
                current.precision,
                current.recall,
                current.f1,
                baseline.mean,
                baseline.std
            ]).reshape(1, -1)
            
            try:
                # Predict degradation probability
                degradation_prob = self.config.ml_model.predict_proba(features)[0][1]  # Probability of degradation
                is_degraded = degradation_prob > 0.5  # Threshold can be configurable
                
                if is_degraded:
                    # For ML-based detection, we consider all metrics as potentially affected
                    for metric_name in self._metric_names:
                        current_value = getattr(current, metric_name)
                        detail = DegradationDetails(
                            metric_name=metric_name,
                            current_value=current_value,
                            baseline_value=baseline.mean,
                            deviation=0.0,  # Will be calculated in __post_init__
                            relative_deviation=0.0,  # Will be calculated in __post_init__
                            statistical_significance={
                                "ml_probability": degradation_prob,
                                "model_type": str(type(self.config.ml_model).__name__)
                            }
                        )
                        affected_metrics.append(detail)
                
                metadata = {
                    "ml_model_type": str(type(self.config.ml_model).__name__),
                    "degradation_probability": degradation_prob,
                    "prediction_threshold": 0.5
                }
            except Exception as e:
                # Fallback to simple threshold if ML model fails
                return self._simple_threshold_detection(current, baseline)
        else:
            # No ML model provided, use simple threshold as fallback
            return self._simple_threshold_detection(current, baseline)
        
        return DegradationResult(
            degrade_flag=len(affected_metrics) > 0,
            affected_metrics=affected_metrics,
            detection_algorithm=self.config.algorithm,
            metadata=metadata
        )

