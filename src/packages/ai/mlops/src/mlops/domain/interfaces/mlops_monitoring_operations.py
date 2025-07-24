"""Domain interfaces for MLOps monitoring operations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from enum import Enum


class MonitoringAlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MonitoringAlertStatus(Enum):
    """Alert status states."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class ModelHealthStatus(Enum):
    """Model health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class DataDriftType(Enum):
    """Types of data drift."""
    FEATURE_DRIFT = "feature_drift"
    TARGET_DRIFT = "target_drift"
    PREDICTION_DRIFT = "prediction_drift"
    SCHEMA_DRIFT = "schema_drift"


@dataclass
class MetricThreshold:
    """Metric threshold configuration."""
    metric_name: str
    operator: str  # >, <, >=, <=, ==, !=
    threshold_value: float
    severity: MonitoringAlertSeverity
    description: str


@dataclass
class MonitoringRule:
    """Monitoring rule definition."""
    rule_id: str
    name: str
    description: str
    model_id: Optional[str]
    deployment_id: Optional[str]
    thresholds: List[MetricThreshold]
    evaluation_window: timedelta
    alert_frequency: timedelta
    enabled: bool
    tags: List[str]


@dataclass
class PerformanceMetrics:
    """Model performance metrics."""
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    auc_roc: Optional[float] = None
    latency_p50: Optional[float] = None
    latency_p95: Optional[float] = None
    latency_p99: Optional[float] = None
    throughput: Optional[float] = None
    error_rate: Optional[float] = None
    custom_metrics: Optional[Dict[str, float]] = None
    timestamp: Optional[datetime] = None


@dataclass
class InfrastructureMetrics:
    """Infrastructure performance metrics."""
    cpu_usage: Optional[float] = None
    memory_usage: Optional[float] = None
    gpu_usage: Optional[float] = None
    disk_usage: Optional[float] = None
    network_io: Optional[float] = None
    request_queue_size: Optional[int] = None
    active_connections: Optional[int] = None
    replica_count: Optional[int] = None
    custom_metrics: Optional[Dict[str, float]] = None
    timestamp: Optional[datetime] = None


@dataclass
class DataQualityMetrics:
    """Data quality metrics."""
    missing_values_ratio: Optional[float] = None
    outlier_ratio: Optional[float] = None
    schema_violations: Optional[int] = None
    data_freshness: Optional[timedelta] = None
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    duplicate_ratio: Optional[float] = None
    data_type_violations: Optional[int] = None
    custom_checks: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None


@dataclass
class DriftDetectionResult:
    """Data drift detection result."""
    drift_type: DataDriftType
    is_drift_detected: bool
    drift_score: float
    confidence: float
    affected_features: List[str]
    reference_period: Dict[str, datetime]
    detection_period: Dict[str, datetime]
    statistical_tests: Dict[str, Any]
    visualizations: Optional[Dict[str, str]] = None
    timestamp: datetime = None


@dataclass
class MonitoringAlert:
    """Monitoring alert."""
    alert_id: str
    rule_id: str
    severity: MonitoringAlertSeverity
    status: MonitoringAlertStatus
    title: str
    description: str
    model_id: Optional[str]
    deployment_id: Optional[str]
    triggered_at: datetime
    acknowledged_at: Optional[datetime]
    resolved_at: Optional[datetime]
    metadata: Dict[str, Any]
    remediation_suggestions: List[str]


@dataclass
class ModelHealthReport:
    """Comprehensive model health report."""
    model_id: str
    deployment_id: Optional[str]
    overall_health: ModelHealthStatus
    performance_health: ModelHealthStatus
    infrastructure_health: ModelHealthStatus
    data_quality_health: ModelHealthStatus
    drift_health: ModelHealthStatus
    recent_alerts: List[MonitoringAlert]
    performance_metrics: PerformanceMetrics
    infrastructure_metrics: InfrastructureMetrics
    data_quality_metrics: DataQualityMetrics
    drift_results: List[DriftDetectionResult]
    recommendations: List[str]
    report_timestamp: datetime


class ModelPerformanceMonitoringPort(ABC):
    """Port for model performance monitoring operations."""

    @abstractmethod
    async def log_prediction_metrics(
        self,
        model_id: str,
        deployment_id: Optional[str],
        metrics: PerformanceMetrics
    ) -> None:
        """Log model prediction performance metrics.
        
        Args:
            model_id: ID of the model
            deployment_id: ID of the deployment (optional)
            metrics: Performance metrics to log
        """
        pass

    @abstractmethod
    async def get_performance_metrics(
        self,
        model_id: str,
        deployment_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        aggregation_window: Optional[timedelta] = None
    ) -> List[PerformanceMetrics]:
        """Get historical performance metrics.
        
        Args:
            model_id: ID of the model
            deployment_id: ID of the deployment (optional)
            start_time: Start time for metrics query
            end_time: End time for metrics query
            aggregation_window: Time window for aggregation
            
        Returns:
            List of performance metrics
        """
        pass

    @abstractmethod
    async def calculate_performance_degradation(
        self,
        model_id: str,
        deployment_id: Optional[str] = None,
        baseline_period: timedelta = timedelta(days=7),
        comparison_period: timedelta = timedelta(days=1)
    ) -> Dict[str, float]:
        """Calculate performance degradation compared to baseline.
        
        Args:
            model_id: ID of the model
            deployment_id: ID of the deployment (optional)
            baseline_period: Baseline period for comparison
            comparison_period: Recent period for comparison
            
        Returns:
            Performance degradation metrics
        """
        pass

    @abstractmethod
    async def detect_performance_anomalies(
        self,
        model_id: str,
        deployment_id: Optional[str] = None,
        lookback_window: timedelta = timedelta(hours=24),
        sensitivity: float = 0.95
    ) -> List[Dict[str, Any]]:
        """Detect performance anomalies.
        
        Args:
            model_id: ID of the model
            deployment_id: ID of the deployment (optional)
            lookback_window: Time window to analyze
            sensitivity: Anomaly detection sensitivity
            
        Returns:
            List of detected anomalies
        """
        pass


class InfrastructureMonitoringPort(ABC):
    """Port for infrastructure monitoring operations."""

    @abstractmethod
    async def log_infrastructure_metrics(
        self,
        deployment_id: str,
        metrics: InfrastructureMetrics
    ) -> None:
        """Log infrastructure metrics.
        
        Args:
            deployment_id: ID of the deployment
            metrics: Infrastructure metrics to log
        """
        pass

    @abstractmethod
    async def get_infrastructure_metrics(
        self,
        deployment_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        aggregation_window: Optional[timedelta] = None
    ) -> List[InfrastructureMetrics]:
        """Get historical infrastructure metrics.
        
        Args:
            deployment_id: ID of the deployment
            start_time: Start time for metrics query
            end_time: End time for metrics query
            aggregation_window: Time window for aggregation
            
        Returns:
            List of infrastructure metrics
        """
        pass

    @abstractmethod
    async def check_resource_utilization(
        self,
        deployment_id: str,
        threshold_config: Dict[str, float]
    ) -> Dict[str, Any]:
        """Check resource utilization against thresholds.
        
        Args:
            deployment_id: ID of the deployment
            threshold_config: Resource utilization thresholds
            
        Returns:
            Resource utilization status
        """
        pass

    @abstractmethod
    async def predict_resource_needs(
        self,
        deployment_id: str,
        forecast_horizon: timedelta = timedelta(hours=24)
    ) -> Dict[str, Any]:
        """Predict future resource needs.
        
        Args:
            deployment_id: ID of the deployment
            forecast_horizon: Time horizon for prediction
            
        Returns:
            Resource prediction results
        """
        pass


class DataQualityMonitoringPort(ABC):
    """Port for data quality monitoring operations."""

    @abstractmethod
    async def log_data_quality_metrics(
        self,
        model_id: str,
        deployment_id: Optional[str],
        metrics: DataQualityMetrics
    ) -> None:
        """Log data quality metrics.
        
        Args:
            model_id: ID of the model
            deployment_id: ID of the deployment (optional)
            metrics: Data quality metrics to log
        """
        pass

    @abstractmethod
    async def get_data_quality_metrics(
        self,
        model_id: str,
        deployment_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[DataQualityMetrics]:
        """Get historical data quality metrics.
        
        Args:
            model_id: ID of the model
            deployment_id: ID of the deployment (optional)
            start_time: Start time for metrics query
            end_time: End time for metrics query
            
        Returns:
            List of data quality metrics
        """
        pass

    @abstractmethod
    async def validate_input_data(
        self,
        model_id: str,
        data_sample: Dict[str, Any],
        schema_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate input data against expected schema.
        
        Args:
            model_id: ID of the model
            data_sample: Sample of input data
            schema_config: Expected data schema configuration
            
        Returns:
            Validation results
        """
        pass

    @abstractmethod
    async def detect_data_quality_issues(
        self,
        model_id: str,
        deployment_id: Optional[str] = None,
        check_config: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Detect data quality issues.
        
        Args:
            model_id: ID of the model
            deployment_id: ID of the deployment (optional)
            check_config: Data quality check configuration
            
        Returns:
            List of detected data quality issues
        """
        pass


class DataDriftMonitoringPort(ABC):
    """Port for data drift monitoring operations."""

    @abstractmethod
    async def detect_feature_drift(
        self,
        model_id: str,
        reference_data: Dict[str, Any],
        current_data: Dict[str, Any],
        drift_config: Optional[Dict[str, Any]] = None
    ) -> DriftDetectionResult:
        """Detect feature drift.
        
        Args:
            model_id: ID of the model
            reference_data: Reference dataset for comparison
            current_data: Current dataset to check for drift
            drift_config: Drift detection configuration
            
        Returns:
            Drift detection result
        """
        pass

    @abstractmethod
    async def detect_target_drift(
        self,
        model_id: str,
        reference_targets: List[Any],
        current_targets: List[Any],
        drift_config: Optional[Dict[str, Any]] = None
    ) -> DriftDetectionResult:
        """Detect target drift.
        
        Args:
            model_id: ID of the model
            reference_targets: Reference target values
            current_targets: Current target values
            drift_config: Drift detection configuration
            
        Returns:
            Drift detection result
        """
        pass

    @abstractmethod
    async def detect_prediction_drift(
        self,
        model_id: str,
        deployment_id: Optional[str] = None,
        reference_period: timedelta = timedelta(days=7),
        comparison_period: timedelta = timedelta(days=1),
        drift_config: Optional[Dict[str, Any]] = None
    ) -> DriftDetectionResult:
        """Detect prediction drift.
        
        Args:
            model_id: ID of the model
            deployment_id: ID of the deployment (optional)
            reference_period: Reference period for comparison
            comparison_period: Recent period for comparison
            drift_config: Drift detection configuration
            
        Returns:
            Drift detection result
        """
        pass

    @abstractmethod
    async def get_drift_history(
        self,
        model_id: str,
        deployment_id: Optional[str] = None,
        drift_type: Optional[DataDriftType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[DriftDetectionResult]:
        """Get historical drift detection results.
        
        Args:
            model_id: ID of the model
            deployment_id: ID of the deployment (optional)
            drift_type: Type of drift to filter by
            start_time: Start time for query
            end_time: End time for query
            
        Returns:
            List of drift detection results
        """
        pass


class AlertingPort(ABC):
    """Port for alerting operations."""

    @abstractmethod
    async def create_monitoring_rule(self, rule: MonitoringRule) -> str:
        """Create a new monitoring rule.
        
        Args:
            rule: Monitoring rule configuration
            
        Returns:
            Rule ID
        """
        pass

    @abstractmethod
    async def get_monitoring_rule(self, rule_id: str) -> Optional[MonitoringRule]:
        """Get monitoring rule configuration.
        
        Args:
            rule_id: ID of the rule
            
        Returns:
            Monitoring rule or None if not found
        """
        pass

    @abstractmethod
    async def update_monitoring_rule(
        self,
        rule_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Update monitoring rule configuration.
        
        Args:
            rule_id: ID of the rule
            updates: Rule updates
            
        Returns:
            True if update successful
        """
        pass

    @abstractmethod
    async def delete_monitoring_rule(self, rule_id: str) -> bool:
        """Delete monitoring rule.
        
        Args:
            rule_id: ID of the rule
            
        Returns:
            True if deletion successful
        """
        pass

    @abstractmethod
    async def trigger_alert(
        self,
        rule_id: str,
        alert_data: Dict[str, Any]
    ) -> str:
        """Trigger a monitoring alert.
        
        Args:
            rule_id: ID of the triggering rule
            alert_data: Alert data and context
            
        Returns:
            Alert ID
        """
        pass

    @abstractmethod
    async def get_alert(self, alert_id: str) -> Optional[MonitoringAlert]:
        """Get alert information.
        
        Args:
            alert_id: ID of the alert
            
        Returns:
            Alert information or None if not found
        """
        pass

    @abstractmethod
    async def list_alerts(
        self,
        model_id: Optional[str] = None,
        deployment_id: Optional[str] = None,
        severity: Optional[MonitoringAlertSeverity] = None,
        status: Optional[MonitoringAlertStatus] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[MonitoringAlert]:
        """List alerts with optional filters.
        
        Args:
            model_id: Filter by model ID
            deployment_id: Filter by deployment ID
            severity: Filter by severity
            status: Filter by status
            start_time: Start time for query
            end_time: End time for query
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            List of alerts
        """
        pass

    @abstractmethod
    async def acknowledge_alert(
        self,
        alert_id: str,
        acknowledged_by: str,
        acknowledgment_note: Optional[str] = None
    ) -> bool:
        """Acknowledge an alert.
        
        Args:
            alert_id: ID of the alert
            acknowledged_by: User acknowledging the alert
            acknowledgment_note: Optional note
            
        Returns:
            True if acknowledgment successful
        """
        pass

    @abstractmethod
    async def resolve_alert(
        self,
        alert_id: str,
        resolved_by: str,
        resolution_note: Optional[str] = None
    ) -> bool:
        """Resolve an alert.
        
        Args:
            alert_id: ID of the alert
            resolved_by: User resolving the alert
            resolution_note: Optional resolution note
            
        Returns:
            True if resolution successful
        """
        pass


class HealthCheckPort(ABC):
    """Port for health check operations."""

    @abstractmethod
    async def check_model_health(
        self,
        model_id: str,
        deployment_id: Optional[str] = None
    ) -> ModelHealthReport:
        """Perform comprehensive model health check.
        
        Args:
            model_id: ID of the model
            deployment_id: ID of the deployment (optional)
            
        Returns:
            Comprehensive health report
        """
        pass

    @abstractmethod
    async def check_deployment_health(
        self,
        deployment_id: str
    ) -> Dict[str, Any]:
        """Check deployment health status.
        
        Args:
            deployment_id: ID of the deployment
            
        Returns:
            Deployment health status
        """
        pass

    @abstractmethod
    async def run_health_diagnostics(
        self,
        model_id: str,
        deployment_id: Optional[str] = None,
        diagnostic_tests: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Run detailed health diagnostics.
        
        Args:
            model_id: ID of the model
            deployment_id: ID of the deployment (optional)
            diagnostic_tests: Specific tests to run
            
        Returns:
            Diagnostic results
        """
        pass

    @abstractmethod
    async def get_health_history(
        self,
        model_id: str,
        deployment_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[ModelHealthReport]:
        """Get historical health reports.
        
        Args:
            model_id: ID of the model
            deployment_id: ID of the deployment (optional)
            start_time: Start time for query
            end_time: End time for query
            
        Returns:
            List of historical health reports
        """
        pass