"""Anomaly KPI schemas for anomaly detection metrics.

This module provides Pydantic schemas for anomaly detection KPIs including
detection performance metrics, classification metrics, and time series analysis.

Schemas:
    AnomalyKPIFrame: Main anomaly KPI frame
    AnomalyDetectionMetrics: Detection performance metrics
    AnomalyClassificationMetrics: Classification quality metrics
    AnomalyTimeSeriesMetrics: Time series anomaly metrics
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime

from pydantic import BaseModel, Field, validator

from .base import RealTimeMetricFrame, MetricMetadata


class AnomalySeverity(str, Enum):
    """Anomaly severity levels."""
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AnomalyCategory(str, Enum):
    """Anomaly categories."""
    STATISTICAL = "statistical"
    BEHAVIORAL = "behavioral"
    CONTEXTUAL = "contextual"
    COLLECTIVE = "collective"
    TEMPORAL = "temporal"


class AnomalyDetectionMetrics(BaseModel):
    """Core anomaly detection performance metrics."""
    
    accuracy: float = Field(ge=0.0, le=1.0, description="Detection accuracy")
    precision: float = Field(ge=0.0, le=1.0, description="Precision score")
    recall: float = Field(ge=0.0, le=1.0, description="Recall score")
    f1_score: float = Field(ge=0.0, le=1.0, description="F1 score")
    false_positive_rate: float = Field(ge=0.0, le=1.0, description="False positive rate")
    false_negative_rate: float = Field(ge=0.0, le=1.0, description="False negative rate")
    roc_auc: Optional[float] = Field(None, ge=0.0, le=1.0, description="ROC AUC score")
    pr_auc: Optional[float] = Field(None, ge=0.0, le=1.0, description="PR AUC score")
    
    @validator('f1_score')
    def validate_f1_score(cls, v: float, values: Dict[str, Any]) -> float:
        """Validate F1 score consistency with precision and recall."""
        if 'precision' in values and 'recall' in values:
            precision = values['precision']
            recall = values['recall']
            if precision + recall > 0:
                expected_f1 = 2 * (precision * recall) / (precision + recall)
                if abs(v - expected_f1) > 0.01:  # Allow small floating point errors
                    raise ValueError(f"F1 score {v} inconsistent with precision {precision} and recall {recall}")
        return v


class AnomalyClassificationMetrics(BaseModel):
    """Anomaly classification quality metrics."""
    
    true_positives: int = Field(ge=0, description="True positive count")
    false_positives: int = Field(ge=0, description="False positive count")  
    true_negatives: int = Field(ge=0, description="True negative count")
    false_negatives: int = Field(ge=0, description="False negative count")
    
    anomalies_detected: int = Field(ge=0, description="Total anomalies detected")
    anomalies_confirmed: int = Field(ge=0, description="Confirmed anomalies")
    anomalies_dismissed: int = Field(ge=0, description="Dismissed anomalies")
    
    severity_distribution: Dict[AnomalySeverity, int] = Field(
        default_factory=dict,
        description="Distribution of anomalies by severity"
    )
    
    category_distribution: Dict[AnomalyCategory, int] = Field(
        default_factory=dict,
        description="Distribution of anomalies by category"
    )
    
    @validator('anomalies_confirmed')
    def validate_confirmed_anomalies(cls, v: int, values: Dict[str, Any]) -> int:
        """Validate confirmed anomalies don't exceed detected."""
        if 'anomalies_detected' in values and v > values['anomalies_detected']:
            raise ValueError("Confirmed anomalies cannot exceed detected anomalies")
        return v


class AnomalyTimeSeriesMetrics(BaseModel):
    """Time series specific anomaly metrics."""
    
    detection_latency: float = Field(ge=0.0, description="Detection latency in seconds")
    processing_time: float = Field(ge=0.0, description="Processing time in seconds")
    data_freshness: float = Field(ge=0.0, description="Data freshness in seconds")
    
    window_size: int = Field(gt=0, description="Time window size for analysis")
    sample_rate: float = Field(gt=0.0, description="Sample rate in Hz")
    
    trend_anomalies: int = Field(ge=0, description="Trend-based anomalies")
    seasonal_anomalies: int = Field(ge=0, description="Seasonal anomalies")
    point_anomalies: int = Field(ge=0, description="Point anomalies")
    
    drift_detected: bool = Field(default=False, description="Concept drift detected")
    seasonality_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Seasonality score")
    
    @validator('sample_rate')
    def validate_sample_rate(cls, v: float) -> float:
        """Validate sample rate is reasonable."""
        if v > 1000.0:  # 1kHz max
            raise ValueError("Sample rate cannot exceed 1000 Hz")
        return v


class AnomalyKPIFrame(RealTimeMetricFrame):
    """Main anomaly KPI frame containing all anomaly detection metrics."""
    
    # Core detection metrics
    detection_metrics: AnomalyDetectionMetrics
    classification_metrics: AnomalyClassificationMetrics
    time_series_metrics: Optional[AnomalyTimeSeriesMetrics] = None
    
    # Operational metrics
    model_name: str = Field(description="Name of the anomaly detection model")
    model_version: str = Field(description="Version of the model")
    dataset_id: str = Field(description="ID of the dataset being analyzed")
    
    # Performance indicators
    throughput: float = Field(ge=0.0, description="Throughput in samples/second")
    cpu_usage: float = Field(ge=0.0, le=100.0, description="CPU usage percentage")
    memory_usage: float = Field(ge=0.0, description="Memory usage in MB")
    
    # Alert information
    active_alerts: int = Field(ge=0, description="Number of active alerts")
    critical_alerts: int = Field(ge=0, description="Number of critical alerts")
    alert_resolution_time: Optional[float] = Field(None, ge=0.0, description="Average alert resolution time")
    
    # Quality metrics
    confidence_score: float = Field(ge=0.0, le=1.0, description="Overall confidence score")
    data_quality_score: float = Field(ge=0.0, le=1.0, description="Data quality score")
    
    # Additional context
    business_context: Optional[Dict[str, Any]] = Field(None, description="Business context metadata")
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        validate_assignment = True
        extra = "forbid"
        
    @validator('critical_alerts')
    def validate_critical_alerts(cls, v: int, values: Dict[str, Any]) -> int:
        """Validate critical alerts don't exceed active alerts."""
        if 'active_alerts' in values and v > values['active_alerts']:
            raise ValueError("Critical alerts cannot exceed active alerts")
        return v
    
    @validator('memory_usage')
    def validate_memory_usage(cls, v: float) -> float:
        """Validate memory usage is reasonable."""
        if v > 100000.0:  # 100GB max
            raise ValueError("Memory usage seems unreasonably high")
        return v
        
    def get_anomaly_rate(self) -> float:
        """Calculate overall anomaly rate."""
        total_samples = (
            self.classification_metrics.true_positives +
            self.classification_metrics.false_positives +
            self.classification_metrics.true_negatives +
            self.classification_metrics.false_negatives
        )
        
        if total_samples == 0:
            return 0.0
            
        anomalies = (
            self.classification_metrics.true_positives +
            self.classification_metrics.false_negatives
        )
        
        return anomalies / total_samples
    
    def get_performance_summary(self) -> Dict[str, float]:
        """Get a summary of key performance metrics."""
        return {
            "accuracy": self.detection_metrics.accuracy,
            "precision": self.detection_metrics.precision,
            "recall": self.detection_metrics.recall,
            "f1_score": self.detection_metrics.f1_score,
            "anomaly_rate": self.get_anomaly_rate(),
            "confidence_score": self.confidence_score,
            "throughput": self.throughput,
        }
    
    def is_healthy(self) -> bool:
        """Check if the anomaly detection system is healthy."""
        return all([
            self.detection_metrics.accuracy >= 0.8,
            self.detection_metrics.precision >= 0.7,
            self.detection_metrics.recall >= 0.7,
            self.confidence_score >= 0.8,
            self.data_quality_score >= 0.8,
            self.cpu_usage <= 80.0,
            self.memory_usage <= 80000.0,  # 80GB
        ])
