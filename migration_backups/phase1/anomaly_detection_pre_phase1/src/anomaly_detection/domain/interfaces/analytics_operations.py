"""Analytics Operations Interfaces (Ports).

This module defines the abstract interfaces for analytics operations that the
anomaly detection domain requires. These interfaces represent the "ports"
in hexagonal architecture, defining contracts for external analytics services
without coupling to specific implementations.

Following DDD principles, these interfaces belong to the domain layer and
define what the domain needs from external analytics capabilities like
A/B testing, performance analysis, reporting, and business intelligence.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import uuid

from ..entities.detection_result import DetectionResult
from ..entities.model import Model


class TestStatus(Enum):
    """Status of an A/B test."""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    TERMINATED = "terminated"


class TestResult(Enum):
    """Result of an A/B test."""
    INCONCLUSIVE = "inconclusive"
    VARIANT_A_WINS = "variant_a_wins"
    VARIANT_B_WINS = "variant_b_wins"
    NO_SIGNIFICANT_DIFFERENCE = "no_significant_difference"


class StatisticalSignificance(Enum):
    """Level of statistical significance."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class ReportType(Enum):
    """Type of analytics report."""
    PERFORMANCE = "performance"
    COMPARISON = "comparison"
    TREND = "trend"
    SUMMARY = "summary"
    DETAILED = "detailed"
    EXECUTIVE = "executive"


class AlertType(Enum):
    """Type of analytics alert."""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    ANOMALY_SPIKE = "anomaly_spike"
    MODEL_DRIFT = "model_drift"
    BUSINESS_IMPACT = "business_impact"
    SYSTEM_ERROR = "system_error"


@dataclass
class TestVariant:
    """A variant in an A/B test."""
    variant_id: str
    name: str
    model_id: str
    model_version: str
    traffic_percentage: float
    description: str = ""
    configuration: Optional[Dict[str, Any]] = None


@dataclass
class ABTestConfiguration:
    """Configuration for an A/B test."""
    test_id: str
    name: str
    description: str
    variants: List[TestVariant]
    success_metrics: List[str]
    minimum_sample_size: int
    maximum_duration_days: int
    significance_level: float = 0.05
    power: float = 0.8
    traffic_split_type: str = "random"
    exclusion_rules: Optional[Dict[str, Any]] = None


@dataclass
class TestResults:
    """Results of an A/B test."""
    test_id: str
    status: TestStatus
    result: TestResult
    confidence_level: float
    p_value: float
    effect_size: float
    sample_sizes: Dict[str, int]
    conversion_rates: Dict[str, float]
    statistical_significance: StatisticalSignificance
    recommendations: List[str]
    detailed_metrics: Dict[str, Any]


@dataclass
class PerformanceAnalysis:
    """Analysis of model performance over time."""
    model_id: str
    analysis_period: Tuple[datetime, datetime]
    accuracy_trend: List[Tuple[datetime, float]]
    precision_trend: List[Tuple[datetime, float]]
    recall_trend: List[Tuple[datetime, float]]
    f1_trend: List[Tuple[datetime, float]]
    throughput_trend: List[Tuple[datetime, float]]
    latency_trend: List[Tuple[datetime, float]]
    error_rate_trend: List[Tuple[datetime, float]]
    anomaly_detection_rate: List[Tuple[datetime, float]]
    insights: List[str]
    recommendations: List[str]


@dataclass
class BusinessImpactMetrics:
    """Business impact metrics for anomaly detection."""
    total_anomalies_detected: int
    false_positive_count: int
    false_negative_count: int
    true_positive_count: int
    cost_savings_estimate: float
    time_to_detection_minutes: float
    investigation_time_minutes: float
    resolution_time_minutes: float
    customer_impact_score: float
    business_value_score: float


@dataclass
class AlertConfiguration:
    """Configuration for an analytics alert."""
    alert_id: str
    name: str
    description: str
    alert_type: AlertType
    conditions: Dict[str, Any]
    thresholds: Dict[str, float]
    notification_channels: List[str]
    escalation_rules: Optional[Dict[str, Any]] = None
    active: bool = True


class AnalyticsABTestingPort(ABC):
    """Port for A/B testing operations.
    
    This interface defines the contract for setting up, running, and
    analyzing A/B tests for anomaly detection models.
    """

    @abstractmethod
    async def create_ab_test(
        self,
        configuration: ABTestConfiguration,
        created_by: str = "system"
    ) -> str:
        """Create a new A/B test.
        
        Args:
            configuration: Test configuration
            created_by: User creating the test
            
        Returns:
            Unique test identifier
            
        Raises:
            ABTestCreationError: If test creation fails
        """
        pass

    @abstractmethod
    async def start_ab_test(self, test_id: str) -> None:
        """Start an A/B test.
        
        Args:
            test_id: Test identifier
            
        Raises:
            ABTestStartError: If test start fails
        """
        pass

    @abstractmethod
    async def stop_ab_test(self, test_id: str, reason: str = "") -> None:
        """Stop an A/B test.
        
        Args:
            test_id: Test identifier
            reason: Optional reason for stopping
            
        Raises:
            ABTestStopError: If test stop fails
        """
        pass

    @abstractmethod
    async def record_test_interaction(
        self,
        test_id: str,
        variant_id: str,
        user_id: str,
        interaction_data: Dict[str, Any],
        outcome_metrics: Dict[str, float]
    ) -> None:
        """Record an interaction with a test variant.
        
        Args:
            test_id: Test identifier
            variant_id: Variant identifier
            user_id: User identifier
            interaction_data: Data about the interaction
            outcome_metrics: Measured outcomes
            
        Raises:
            InteractionRecordingError: If recording fails
        """
        pass

    @abstractmethod
    async def get_ab_test_results(self, test_id: str) -> TestResults:
        """Get results of an A/B test.
        
        Args:
            test_id: Test identifier
            
        Returns:
            Test results with statistical analysis
            
        Raises:
            TestResultsError: If results retrieval fails
        """
        pass

    @abstractmethod
    async def analyze_test_significance(self, test_id: str) -> Dict[str, Any]:
        """Analyze statistical significance of test results.
        
        Args:
            test_id: Test identifier
            
        Returns:
            Statistical analysis results
            
        Raises:
            StatisticalAnalysisError: If analysis fails
        """
        pass

    @abstractmethod
    async def get_traffic_allocation(
        self,
        test_id: str,
        user_id: str,
        context: Dict[str, Any]
    ) -> str:
        """Get traffic allocation for a user in an A/B test.
        
        Args:
            test_id: Test identifier
            user_id: User identifier
            context: Additional context for allocation
            
        Returns:
            Variant identifier for the user
            
        Raises:
            TrafficAllocationError: If allocation fails
        """
        pass

    @abstractmethod
    async def list_active_tests(self) -> List[ABTestConfiguration]:
        """List all active A/B tests.
        
        Returns:
            List of active test configurations
            
        Raises:
            TestListingError: If listing fails
        """
        pass


class AnalyticsPerformancePort(ABC):
    """Port for performance analytics operations.
    
    This interface defines the contract for analyzing and monitoring
    the performance of anomaly detection models and systems.
    """

    @abstractmethod
    async def analyze_model_performance(
        self,
        model_id: str,
        start_time: datetime,
        end_time: datetime,
        metrics: Optional[List[str]] = None
    ) -> PerformanceAnalysis:
        """Analyze model performance over a time period.
        
        Args:
            model_id: Model identifier
            start_time: Start of analysis period
            end_time: End of analysis period
            metrics: Optional specific metrics to analyze
            
        Returns:
            Performance analysis results
            
        Raises:
            PerformanceAnalysisError: If analysis fails
        """
        pass

    @abstractmethod
    async def compare_model_performance(
        self,
        model_ids: List[str],
        start_time: datetime,
        end_time: datetime,
        comparison_metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Compare performance of multiple models.
        
        Args:
            model_ids: List of model identifiers
            start_time: Start of comparison period
            end_time: End of comparison period
            comparison_metrics: Optional specific metrics to compare
            
        Returns:
            Comparison results with rankings and insights
            
        Raises:
            ModelComparisonError: If comparison fails
        """
        pass

    @abstractmethod
    async def calculate_business_impact(
        self,
        model_id: str,
        start_time: datetime,
        end_time: datetime,
        business_context: Dict[str, Any]
    ) -> BusinessImpactMetrics:
        """Calculate business impact of anomaly detection.
        
        Args:
            model_id: Model identifier
            start_time: Start of analysis period
            end_time: End of analysis period
            business_context: Business context for impact calculation
            
        Returns:
            Business impact metrics
            
        Raises:
            BusinessImpactError: If calculation fails
        """
        pass

    @abstractmethod
    async def detect_performance_degradation(
        self,
        model_id: str,
        baseline_period: Tuple[datetime, datetime],
        current_period: Tuple[datetime, datetime],
        sensitivity: float = 0.05
    ) -> Dict[str, Any]:
        """Detect performance degradation by comparing periods.
        
        Args:
            model_id: Model identifier
            baseline_period: Baseline time period
            current_period: Current time period to compare
            sensitivity: Sensitivity threshold for degradation detection
            
        Returns:
            Degradation analysis results
            
        Raises:
            DegradationDetectionError: If detection fails
        """
        pass

    @abstractmethod
    async def generate_performance_forecast(
        self,
        model_id: str,
        historical_data_period: Tuple[datetime, datetime],
        forecast_horizon_days: int
    ) -> Dict[str, Any]:
        """Generate performance forecast for a model.
        
        Args:
            model_id: Model identifier
            historical_data_period: Period of historical data to use
            forecast_horizon_days: Number of days to forecast
            
        Returns:
            Performance forecast with confidence intervals
            
        Raises:
            ForecastGenerationError: If forecast generation fails
        """
        pass

    @abstractmethod
    async def identify_performance_patterns(
        self,
        model_id: str,
        analysis_period: Tuple[datetime, datetime]
    ) -> Dict[str, Any]:
        """Identify patterns in model performance.
        
        Args:
            model_id: Model identifier
            analysis_period: Time period for pattern analysis
            
        Returns:
            Identified patterns and insights
            
        Raises:
            PatternAnalysisError: If pattern analysis fails
        """
        pass


class AnalyticsReportingPort(ABC):
    """Port for analytics reporting operations.
    
    This interface defines the contract for generating various types
    of analytics reports and dashboards.
    """

    @abstractmethod
    async def generate_report(
        self,
        report_type: ReportType,
        report_config: Dict[str, Any],
        output_format: str = "json"
    ) -> Dict[str, Any]:
        """Generate an analytics report.
        
        Args:
            report_type: Type of report to generate
            report_config: Configuration for the report
            output_format: Output format (json, pdf, html, csv)
            
        Returns:
            Generated report data
            
        Raises:
            ReportGenerationError: If report generation fails
        """
        pass

    @abstractmethod
    async def create_dashboard(
        self,
        dashboard_name: str,
        widgets: List[Dict[str, Any]],
        layout: Dict[str, Any],
        access_permissions: Optional[List[str]] = None
    ) -> str:
        """Create an analytics dashboard.
        
        Args:
            dashboard_name: Name of the dashboard
            widgets: List of widget configurations
            layout: Dashboard layout configuration
            access_permissions: Optional access permissions
            
        Returns:
            Dashboard identifier or URL
            
        Raises:
            DashboardCreationError: If dashboard creation fails
        """
        pass

    @abstractmethod
    async def schedule_report(
        self,
        report_config: Dict[str, Any],
        schedule: str,
        recipients: List[str],
        delivery_method: str = "email"
    ) -> str:
        """Schedule recurring report generation.
        
        Args:
            report_config: Report configuration
            schedule: Cron-style schedule expression
            recipients: List of report recipients
            delivery_method: Method for delivering reports
            
        Returns:
            Schedule identifier
            
        Raises:
            ReportSchedulingError: If scheduling fails
        """
        pass

    @abstractmethod
    async def export_data(
        self,
        data_source: str,
        filters: Dict[str, Any],
        export_format: str,
        destination: str
    ) -> str:
        """Export analytics data.
        
        Args:
            data_source: Source of data to export
            filters: Filters to apply to the data
            export_format: Format for export (csv, json, parquet)
            destination: Destination for exported data
            
        Returns:
            Export job identifier
            
        Raises:
            DataExportError: If export fails
        """
        pass


class AnalyticsAlertingPort(ABC):
    """Port for analytics alerting operations.
    
    This interface defines the contract for setting up and managing
    alerts based on analytics insights and thresholds.
    """

    @abstractmethod
    async def create_alert(
        self,
        configuration: AlertConfiguration,
        created_by: str = "system"
    ) -> str:
        """Create an analytics alert.
        
        Args:
            configuration: Alert configuration
            created_by: User creating the alert
            
        Returns:
            Alert identifier
            
        Raises:
            AlertCreationError: If alert creation fails
        """
        pass

    @abstractmethod
    async def evaluate_alert_conditions(
        self,
        alert_id: str,
        current_data: Dict[str, Any]
    ) -> bool:
        """Evaluate if alert conditions are met.
        
        Args:
            alert_id: Alert identifier
            current_data: Current data to evaluate
            
        Returns:
            True if alert should be triggered, False otherwise
            
        Raises:
            AlertEvaluationError: If evaluation fails
        """
        pass

    @abstractmethod
    async def trigger_alert(
        self,
        alert_id: str,
        trigger_data: Dict[str, Any],
        severity: str = "medium"
    ) -> None:
        """Trigger an alert.
        
        Args:
            alert_id: Alert identifier
            trigger_data: Data that triggered the alert
            severity: Alert severity level
            
        Raises:
            AlertTriggerError: If alert trigger fails
        """
        pass

    @abstractmethod
    async def acknowledge_alert(
        self,
        alert_id: str,
        acknowledged_by: str,
        acknowledgment_note: str = ""
    ) -> None:
        """Acknowledge an alert.
        
        Args:
            alert_id: Alert identifier
            acknowledged_by: User acknowledging the alert
            acknowledgment_note: Optional acknowledgment note
            
        Raises:
            AlertAcknowledgmentError: If acknowledgment fails
        """
        pass

    @abstractmethod
    async def get_active_alerts(
        self,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Get list of active alerts.
        
        Args:
            filters: Optional filters for alerts
            
        Returns:
            List of active alerts
            
        Raises:
            AlertListingError: If listing fails
        """
        pass

    @abstractmethod
    async def update_alert_configuration(
        self,
        alert_id: str,
        configuration: AlertConfiguration
    ) -> None:
        """Update alert configuration.
        
        Args:
            alert_id: Alert identifier
            configuration: New alert configuration
            
        Raises:
            AlertUpdateError: If update fails
        """
        pass


# Custom exceptions for analytics operations
class AnalyticsOperationError(Exception):
    """Base exception for analytics operation errors."""
    pass


class ABTestCreationError(AnalyticsOperationError):
    """Exception raised during A/B test creation."""
    pass


class ABTestStartError(AnalyticsOperationError):
    """Exception raised during A/B test start."""
    pass


class ABTestStopError(AnalyticsOperationError):
    """Exception raised during A/B test stop."""
    pass


class InteractionRecordingError(AnalyticsOperationError):
    """Exception raised during interaction recording."""
    pass


class TestResultsError(AnalyticsOperationError):
    """Exception raised during test results retrieval."""
    pass


class StatisticalAnalysisError(AnalyticsOperationError):
    """Exception raised during statistical analysis."""
    pass


class TrafficAllocationError(AnalyticsOperationError):
    """Exception raised during traffic allocation."""
    pass


class TestListingError(AnalyticsOperationError):
    """Exception raised during test listing."""
    pass


class PerformanceAnalysisError(AnalyticsOperationError):
    """Exception raised during performance analysis."""
    pass


class ModelComparisonError(AnalyticsOperationError):
    """Exception raised during model comparison."""
    pass


class BusinessImpactError(AnalyticsOperationError):
    """Exception raised during business impact calculation."""
    pass


class DegradationDetectionError(AnalyticsOperationError):
    """Exception raised during degradation detection."""
    pass


class ForecastGenerationError(AnalyticsOperationError):
    """Exception raised during forecast generation."""
    pass


class PatternAnalysisError(AnalyticsOperationError):
    """Exception raised during pattern analysis."""
    pass


class ReportGenerationError(AnalyticsOperationError):
    """Exception raised during report generation."""
    pass


class DashboardCreationError(AnalyticsOperationError):
    """Exception raised during dashboard creation."""
    pass


class ReportSchedulingError(AnalyticsOperationError):
    """Exception raised during report scheduling."""
    pass


class DataExportError(AnalyticsOperationError):
    """Exception raised during data export."""
    pass


class AlertCreationError(AnalyticsOperationError):
    """Exception raised during alert creation."""
    pass


class AlertEvaluationError(AnalyticsOperationError):
    """Exception raised during alert evaluation."""
    pass


class AlertTriggerError(AnalyticsOperationError):
    """Exception raised during alert trigger."""
    pass


class AlertAcknowledgmentError(AnalyticsOperationError):
    """Exception raised during alert acknowledgment."""
    pass


class AlertListingError(AnalyticsOperationError):
    """Exception raised during alert listing."""
    pass


class AlertUpdateError(AnalyticsOperationError):
    """Exception raised during alert update."""
    pass