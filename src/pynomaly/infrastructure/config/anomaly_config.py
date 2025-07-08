"""Configuration models for anomaly detection system.

This module provides Pydantic models for managing configuration of:
- Default thresholds per metric
- Detection algorithm selection
- Alert escalation rules
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class ComparisonOperator(str, Enum):
    """Supported comparison operators for thresholds."""
    
    GREATER_THAN = ">"
    LESS_THAN = "<"
    GREATER_EQUAL = ">="
    LESS_EQUAL = "<="
    EQUAL = "=="
    NOT_EQUAL = "!="


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class NotificationChannel(str, Enum):
    """Available notification channels."""
    
    EMAIL = "email"
    SLACK = "slack"
    TEAMS = "teams"
    WEBHOOK = "webhook"
    PAGERDUTY = "pagerduty"
    SMS = "sms"
    DASHBOARD = "dashboard"


class MetricThreshold(BaseModel):
    """Configuration for a metric threshold."""
    
    name: str = Field(..., description="Name of the metric")
    value: float = Field(..., description="Threshold value")
    operator: ComparisonOperator = Field(..., description="Comparison operator")
    severity: AlertSeverity = Field(default=AlertSeverity.WARNING, description="Alert severity")
    enabled: bool = Field(default=True, description="Whether threshold is enabled")
    description: Optional[str] = Field(None, description="Description of the threshold")
    
    @field_validator('value')
    @classmethod
    def validate_value(cls, v: float) -> float:
        """Validate threshold value."""
        if v < 0:
            raise ValueError("Threshold value must be non-negative")
        return v


class AlgorithmConfig(BaseModel):
    """Configuration for detection algorithms."""
    
    name: str = Field(..., description="Algorithm name")
    display_name: str = Field(..., description="Human-readable name")
    description: Optional[str] = Field(None, description="Algorithm description")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Default parameters")
    enabled: bool = Field(default=True, description="Whether algorithm is enabled")
    category: str = Field(default="general", description="Algorithm category")
    performance_profile: Dict[str, Any] = Field(default_factory=dict, description="Performance characteristics")
    
    @field_validator('parameters')
    @classmethod
    def validate_parameters(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate algorithm parameters."""
        # Ensure contamination rate is within valid range
        if 'contamination' in v:
            contamination = v['contamination']
            if not (0.0 < contamination <= 0.5):
                raise ValueError("Contamination rate must be between 0.0 and 0.5")
        return v


class EscalationRule(BaseModel):
    """Configuration for alert escalation."""
    
    level: int = Field(..., description="Escalation level (1=first, 2=second, etc.)")
    delay_minutes: int = Field(default=30, description="Delay before escalation in minutes")
    channels: List[NotificationChannel] = Field(..., description="Notification channels")
    recipients: List[str] = Field(default_factory=list, description="Specific recipients")
    conditions: Dict[str, Any] = Field(default_factory=dict, description="Additional conditions")
    
    @field_validator('level')
    @classmethod
    def validate_level(cls, v: int) -> int:
        """Validate escalation level."""
        if v < 1:
            raise ValueError("Escalation level must be at least 1")
        return v
    
    @field_validator('delay_minutes')
    @classmethod
    def validate_delay(cls, v: int) -> int:
        """Validate delay minutes."""
        if v < 0:
            raise ValueError("Delay minutes must be non-negative")
        return v


class AlertRuleConfig(BaseModel):
    """Configuration for alert rules."""
    
    id: str = Field(..., description="Unique rule identifier")
    name: str = Field(..., description="Rule name")
    description: Optional[str] = Field(None, description="Rule description")
    metric_name: str = Field(..., description="Metric to monitor")
    threshold: MetricThreshold = Field(..., description="Threshold configuration")
    escalation_rules: List[EscalationRule] = Field(default_factory=list, description="Escalation rules")
    cooldown_minutes: int = Field(default=15, description="Cooldown period in minutes")
    evaluation_window_minutes: int = Field(default=5, description="Evaluation window in minutes")
    business_hours_only: bool = Field(default=False, description="Only alert during business hours")
    environment_filters: List[str] = Field(default_factory=list, description="Environment filters")
    enabled: bool = Field(default=True, description="Whether rule is enabled")
    
    @field_validator('cooldown_minutes', 'evaluation_window_minutes')
    @classmethod
    def validate_minutes(cls, v: int) -> int:
        """Validate minute values."""
        if v < 1:
            raise ValueError("Minutes must be at least 1")
        return v


class AnomalyDetectionConfig(BaseModel):
    """Main configuration for anomaly detection system."""
    
    # Default thresholds per metric
    default_thresholds: Dict[str, MetricThreshold] = Field(
        default_factory=dict,
        description="Default thresholds for various metrics"
    )
    
    # Detection algorithm configuration
    algorithms: Dict[str, AlgorithmConfig] = Field(
        default_factory=dict,
        description="Available detection algorithms"
    )
    
    default_algorithm: str = Field(
        default="IsolationForest",
        description="Default algorithm to use"
    )
    
    # Alert rules configuration
    alert_rules: Dict[str, AlertRuleConfig] = Field(
        default_factory=dict,
        description="Alert rules configuration"
    )
    
    # Global settings
    global_settings: Dict[str, Any] = Field(
        default_factory=dict,
        description="Global configuration settings"
    )
    
    @field_validator('default_algorithm')
    @classmethod
    def validate_default_algorithm(cls, v: str, values) -> str:
        """Validate default algorithm exists."""
        # Note: We can't validate against algorithms here due to validation order
        # This will be validated at runtime
        return v


def create_default_config() -> AnomalyDetectionConfig:
    """Create default configuration for anomaly detection system."""
    
    # Default thresholds
    default_thresholds = {
        "error_rate": MetricThreshold(
            name="error_rate",
            value=5.0,
            operator=ComparisonOperator.GREATER_THAN,
            severity=AlertSeverity.ERROR,
            description="Error rate percentage threshold"
        ),
        "avg_detection_time": MetricThreshold(
            name="avg_detection_time",
            value=60.0,
            operator=ComparisonOperator.GREATER_THAN,
            severity=AlertSeverity.CRITICAL,
            description="Average detection time in seconds"
        ),
        "memory_usage": MetricThreshold(
            name="memory_usage",
            value=80.0,
            operator=ComparisonOperator.GREATER_THAN,
            severity=AlertSeverity.WARNING,
            description="Memory usage percentage threshold"
        ),
        "cpu_usage": MetricThreshold(
            name="cpu_usage",
            value=85.0,
            operator=ComparisonOperator.GREATER_THAN,
            severity=AlertSeverity.WARNING,
            description="CPU usage percentage threshold"
        ),
        "anomaly_score": MetricThreshold(
            name="anomaly_score",
            value=0.8,
            operator=ComparisonOperator.GREATER_THAN,
            severity=AlertSeverity.WARNING,
            description="Anomaly score threshold"
        ),
        "data_quality_score": MetricThreshold(
            name="data_quality_score",
            value=85.0,
            operator=ComparisonOperator.LESS_THAN,
            severity=AlertSeverity.WARNING,
            description="Data quality score threshold"
        ),
        "model_drift": MetricThreshold(
            name="model_drift",
            value=0.3,
            operator=ComparisonOperator.GREATER_THAN,
            severity=AlertSeverity.ERROR,
            description="Model drift detection threshold"
        ),
        "false_positive_rate": MetricThreshold(
            name="false_positive_rate",
            value=10.0,
            operator=ComparisonOperator.GREATER_THAN,
            severity=AlertSeverity.WARNING,
            description="False positive rate percentage threshold"
        ),
    }
    
    # Default algorithms
    algorithms = {
        "IsolationForest": AlgorithmConfig(
            name="IsolationForest",
            display_name="Isolation Forest",
            description="Isolation Forest for anomaly detection",
            parameters={
                "contamination": 0.1,
                "n_estimators": 100,
                "max_samples": "auto",
                "max_features": 1.0
            },
            category="ensemble",
            performance_profile={
                "speed": "fast",
                "memory": "low",
                "accuracy": "high"
            }
        ),
        "LOF": AlgorithmConfig(
            name="LOF",
            display_name="Local Outlier Factor",
            description="Local Outlier Factor for anomaly detection",
            parameters={
                "contamination": 0.1,
                "n_neighbors": 20,
                "algorithm": "auto"
            },
            category="proximity",
            performance_profile={
                "speed": "medium",
                "memory": "medium",
                "accuracy": "high"
            }
        ),
        "OneClassSVM": AlgorithmConfig(
            name="OneClassSVM",
            display_name="One-Class SVM",
            description="One-Class Support Vector Machine for anomaly detection",
            parameters={
                "contamination": 0.1,
                "kernel": "rbf",
                "gamma": "scale",
                "nu": 0.5
            },
            category="boundary",
            performance_profile={
                "speed": "slow",
                "memory": "high",
                "accuracy": "high"
            }
        ),
        "PCA": AlgorithmConfig(
            name="PCA",
            display_name="Principal Component Analysis",
            description="PCA-based anomaly detection",
            parameters={
                "contamination": 0.1,
                "n_components": None,
                "whiten": True
            },
            category="linear",
            performance_profile={
                "speed": "fast",
                "memory": "low",
                "accuracy": "medium"
            }
        ),
        "COPOD": AlgorithmConfig(
            name="COPOD",
            display_name="Copula-based Outlier Detection",
            description="Copula-based outlier detection",
            parameters={
                "contamination": 0.1
            },
            category="probabilistic",
            performance_profile={
                "speed": "fast",
                "memory": "low",
                "accuracy": "medium"
            }
        ),
    }
    
    # Default alert rules
    alert_rules = {
        "high_error_rate": AlertRuleConfig(
            id="high_error_rate",
            name="High Error Rate",
            description="Alert when error rate exceeds threshold",
            metric_name="error_rate",
            threshold=default_thresholds["error_rate"],
            escalation_rules=[
                EscalationRule(
                    level=1,
                    delay_minutes=0,
                    channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK],
                    recipients=["alerts@company.com"]
                ),
                EscalationRule(
                    level=2,
                    delay_minutes=30,
                    channels=[NotificationChannel.EMAIL, NotificationChannel.PAGERDUTY],
                    recipients=["oncall@company.com"]
                )
            ],
            cooldown_minutes=15,
            evaluation_window_minutes=5
        ),
        "critical_performance": AlertRuleConfig(
            id="critical_performance",
            name="Critical Performance Degradation",
            description="Alert when detection time is critically high",
            metric_name="avg_detection_time",
            threshold=default_thresholds["avg_detection_time"],
            escalation_rules=[
                EscalationRule(
                    level=1,
                    delay_minutes=0,
                    channels=[NotificationChannel.EMAIL, NotificationChannel.PAGERDUTY],
                    recipients=["oncall@company.com"]
                )
            ],
            cooldown_minutes=10,
            evaluation_window_minutes=3
        ),
        "model_drift_detected": AlertRuleConfig(
            id="model_drift_detected",
            name="Model Drift Detected",
            description="Alert when model drift is detected",
            metric_name="model_drift",
            threshold=default_thresholds["model_drift"],
            escalation_rules=[
                EscalationRule(
                    level=1,
                    delay_minutes=0,
                    channels=[NotificationChannel.EMAIL, NotificationChannel.TEAMS],
                    recipients=["ml-team@company.com"]
                )
            ],
            cooldown_minutes=60,
            evaluation_window_minutes=15
        ),
    }
    
    # Global settings
    global_settings = {
        "max_alerts_per_hour": 100,
        "correlation_window_minutes": 10,
        "enable_alert_correlation": True,
        "enable_intelligent_suppression": True,
        "business_hours_start": 9,
        "business_hours_end": 17,
        "business_days": ["monday", "tuesday", "wednesday", "thursday", "friday"],
        "timezone": "UTC",
        "notification_retry_attempts": 3,
        "notification_retry_delay": 300,
        "alert_history_retention_days": 30,
        "config_backup_enabled": True,
        "config_backup_interval_hours": 24,
    }
    
    return AnomalyDetectionConfig(
        default_thresholds=default_thresholds,
        algorithms=algorithms,
        default_algorithm="IsolationForest",
        alert_rules=alert_rules,
        global_settings=global_settings
    )
