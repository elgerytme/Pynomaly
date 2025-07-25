"""Domain interfaces for quality assessment operations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Callable
from enum import Enum

from data_quality.domain.entities.data_quality_rule import DataQualityRule, RuleCondition, RuleResult
from data_quality.domain.entities.data_quality_check import DataQualityCheck, DataQualityResult


class RuleType(Enum):
    """Types of data quality rules."""
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    VALIDITY = "validity"
    UNIQUENESS = "uniqueness"
    TIMELINESS = "timeliness"
    INTEGRITY = "integrity"
    CUSTOM = "custom"


class MetricType(Enum):
    """Types of quality metrics."""
    COUNT = "count"
    PERCENTAGE = "percentage"
    RATIO = "ratio"
    SCORE = "score"
    THRESHOLD = "threshold"
    DISTRIBUTION = "distribution"


class AnomalyType(Enum):
    """Types of anomalies."""
    STATISTICAL = "statistical"
    PATTERN = "pattern"
    VOLUME = "volume"
    SCHEMA = "schema"
    TEMPORAL = "temporal"
    BUSINESS_RULE = "business_rule"


@dataclass
class QualityMetric:
    """Quality metric definition."""
    name: str
    metric_type: MetricType
    value: Union[int, float, str]
    threshold: Optional[Union[int, float]] = None
    unit: Optional[str] = None
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AnomalyDetectionConfig:
    """Configuration for anomaly detection."""
    anomaly_types: List[AnomalyType]
    sensitivity: float = 0.95
    window_size: Optional[int] = None
    baseline_period: Optional[int] = None
    custom_rules: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class QualityAssessmentRequest:
    """Request for quality assessment."""
    data_source: str
    assessment_config: Dict[str, Any]
    rules: Optional[List[DataQualityRule]] = None
    baseline_data: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class RuleEvaluationPort(ABC):
    """Port for rule evaluation operations."""
    
    @abstractmethod
    async def evaluate_rule(
        self, 
        data_source: str, 
        rule: DataQualityRule
    ) -> RuleResult:
        """Evaluate a single data quality rule.
        
        Args:
            data_source: Data source identifier
            rule: Data quality rule to evaluate
            
        Returns:
            Rule evaluation result
        """
        pass
    
    @abstractmethod
    async def evaluate_rule_set(
        self, 
        data_source: str, 
        rules: List[DataQualityRule]
    ) -> List[RuleResult]:
        """Evaluate a set of data quality rules.
        
        Args:
            data_source: Data source identifier
            rules: List of data quality rules
            
        Returns:
            List of rule evaluation results
        """
        pass
    
    @abstractmethod
    async def create_custom_rule(
        self, 
        rule_name: str, 
        rule_logic: Callable[[Any], bool], 
        rule_config: Dict[str, Any]
    ) -> DataQualityRule:
        """Create a custom data quality rule.
        
        Args:
            rule_name: Name of the rule
            rule_logic: Function implementing rule logic
            rule_config: Rule configuration
            
        Returns:
            Created data quality rule
        """
        pass
    
    @abstractmethod
    async def validate_rule_syntax(self, rule: DataQualityRule) -> bool:
        """Validate rule syntax and configuration.
        
        Args:
            rule: Data quality rule to validate
            
        Returns:
            True if rule syntax is valid
        """
        pass
    
    @abstractmethod
    async def optimize_rule_execution(
        self, 
        rules: List[DataQualityRule]
    ) -> List[DataQualityRule]:
        """Optimize rule execution order and configuration.
        
        Args:
            rules: List of data quality rules
            
        Returns:
            Optimized list of rules
        """
        pass
    
    @abstractmethod
    async def get_rule_dependencies(
        self, 
        rule: DataQualityRule
    ) -> List[str]:
        """Get dependencies for a rule.
        
        Args:
            rule: Data quality rule
            
        Returns:
            List of dependency identifiers
        """
        pass


class QualityMetricsPort(ABC):
    """Port for quality metrics operations."""
    
    @abstractmethod
    async def calculate_quality_score(
        self, 
        data_source: str, 
        metrics_config: Dict[str, Any]
    ) -> float:
        """Calculate overall quality score for data.
        
        Args:
            data_source: Data source identifier
            metrics_config: Metrics calculation configuration
            
        Returns:
            Quality score (0.0 to 1.0)
        """
        pass
    
    @abstractmethod
    async def calculate_completeness_metrics(
        self, 
        data_source: str, 
        columns: Optional[List[str]] = None
    ) -> List[QualityMetric]:
        """Calculate data completeness metrics.
        
        Args:
            data_source: Data source identifier
            columns: Optional list of columns to analyze
            
        Returns:
            List of completeness metrics
        """
        pass
    
    @abstractmethod
    async def calculate_accuracy_metrics(
        self, 
        data_source: str, 
        reference_data: Optional[str] = None
    ) -> List[QualityMetric]:
        """Calculate data accuracy metrics.
        
        Args:
            data_source: Data source identifier
            reference_data: Optional reference data for comparison
            
        Returns:
            List of accuracy metrics
        """
        pass
    
    @abstractmethod
    async def calculate_consistency_metrics(
        self, 
        data_source: str, 
        consistency_rules: List[Dict[str, Any]]
    ) -> List[QualityMetric]:
        """Calculate data consistency metrics.
        
        Args:
            data_source: Data source identifier
            consistency_rules: List of consistency rules
            
        Returns:
            List of consistency metrics
        """
        pass
    
    @abstractmethod
    async def calculate_uniqueness_metrics(
        self, 
        data_source: str, 
        key_columns: List[str]
    ) -> List[QualityMetric]:
        """Calculate data uniqueness metrics.
        
        Args:
            data_source: Data source identifier
            key_columns: List of columns to check for uniqueness
            
        Returns:
            List of uniqueness metrics
        """
        pass
    
    @abstractmethod
    async def calculate_validity_metrics(
        self, 
        data_source: str, 
        validation_rules: List[Dict[str, Any]]
    ) -> List[QualityMetric]:
        """Calculate data validity metrics.
        
        Args:
            data_source: Data source identifier
            validation_rules: List of validation rules
            
        Returns:
            List of validity metrics
        """
        pass
    
    @abstractmethod
    async def track_metrics_over_time(
        self, 
        data_source: str, 
        metric_names: List[str], 
        time_period: Dict[str, Any]
    ) -> Dict[str, List[QualityMetric]]:
        """Track quality metrics over time.
        
        Args:
            data_source: Data source identifier
            metric_names: List of metric names to track
            time_period: Time period configuration
            
        Returns:
            Time series of quality metrics
        """
        pass


class AnomalyDetectionPort(ABC):
    """Port for anomaly detection operations."""
    
    @abstractmethod
    async def detect_statistical_anomalies(
        self, 
        data_source: str, 
        config: AnomalyDetectionConfig
    ) -> List[Dict[str, Any]]:
        """Detect statistical anomalies in data.
        
        Args:
            data_source: Data source identifier
            config: Anomaly detection configuration
            
        Returns:
            List of detected anomalies
        """
        pass
    
    @abstractmethod
    async def detect_pattern_anomalies(
        self, 
        data_source: str, 
        pattern_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Detect pattern-based anomalies.
        
        Args:
            data_source: Data source identifier
            pattern_config: Pattern detection configuration
            
        Returns:
            List of detected pattern anomalies
        """
        pass
    
    @abstractmethod
    async def detect_volume_anomalies(
        self, 
        data_source: str, 
        baseline_period: int = 30
    ) -> List[Dict[str, Any]]:
        """Detect volume anomalies (unusual data volumes).
        
        Args:
            data_source: Data source identifier
            baseline_period: Baseline period in days
            
        Returns:
            List of detected volume anomalies
        """
        pass
    
    @abstractmethod
    async def detect_schema_anomalies(
        self, 
        data_source: str, 
        expected_schema: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Detect schema anomalies.
        
        Args:
            data_source: Data source identifier
            expected_schema: Expected data schema
            
        Returns:
            List of detected schema anomalies
        """
        pass
    
    @abstractmethod
    async def detect_temporal_anomalies(
        self, 
        data_source: str, 
        time_column: str, 
        expected_patterns: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Detect temporal anomalies.
        
        Args:
            data_source: Data source identifier
            time_column: Column containing timestamps
            expected_patterns: Expected temporal patterns
            
        Returns:
            List of detected temporal anomalies
        """
        pass
    
    @abstractmethod
    async def create_anomaly_baseline(
        self, 
        data_source: str, 
        baseline_config: Dict[str, Any]
    ) -> str:
        """Create baseline for anomaly detection.
        
        Args:
            data_source: Data source identifier
            baseline_config: Baseline configuration
            
        Returns:
            Baseline identifier
        """
        pass
    
    @abstractmethod
    async def update_anomaly_baseline(
        self, 
        baseline_id: str, 
        new_data_source: str
    ) -> bool:
        """Update anomaly detection baseline.
        
        Args:
            baseline_id: Baseline identifier
            new_data_source: New data to incorporate
            
        Returns:
            True if update successful
        """
        pass


class QualityMonitoringPort(ABC):
    """Port for quality monitoring operations."""
    
    @abstractmethod
    async def create_quality_monitor(
        self, 
        monitor_config: Dict[str, Any]
    ) -> str:
        """Create a quality monitoring configuration.
        
        Args:
            monitor_config: Monitor configuration
            
        Returns:
            Monitor identifier
        """
        pass
    
    @abstractmethod
    async def start_monitoring(
        self, 
        monitor_id: str, 
        data_source: str
    ) -> bool:
        """Start quality monitoring for a data source.
        
        Args:
            monitor_id: Monitor identifier
            data_source: Data source to monitor
            
        Returns:
            True if monitoring started successfully
        """
        pass
    
    @abstractmethod
    async def stop_monitoring(self, monitor_id: str) -> bool:
        """Stop quality monitoring.
        
        Args:
            monitor_id: Monitor identifier
            
        Returns:
            True if monitoring stopped successfully
        """
        pass
    
    @abstractmethod
    async def get_monitoring_status(
        self, 
        monitor_id: str
    ) -> Dict[str, Any]:
        """Get current monitoring status.
        
        Args:
            monitor_id: Monitor identifier
            
        Returns:
            Monitoring status information
        """
        pass
    
    @abstractmethod
    async def configure_quality_alerts(
        self, 
        monitor_id: str, 
        alert_config: Dict[str, Any]
    ) -> bool:
        """Configure quality alerts for monitoring.
        
        Args:
            monitor_id: Monitor identifier
            alert_config: Alert configuration
            
        Returns:
            True if alerts configured successfully
        """
        pass
    
    @abstractmethod
    async def get_quality_trends(
        self, 
        monitor_id: str, 
        time_range: Dict[str, datetime]
    ) -> Dict[str, Any]:
        """Get quality trends over time.
        
        Args:
            monitor_id: Monitor identifier
            time_range: Time range for trend analysis
            
        Returns:
            Quality trend data
        """
        pass


class DataLineagePort(ABC):
    """Port for data lineage operations."""
    
    @abstractmethod
    async def track_data_lineage(
        self, 
        source_data: str, 
        transformation: str, 
        target_data: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Track data lineage for a transformation.
        
        Args:
            source_data: Source data identifier
            transformation: Transformation description
            target_data: Target data identifier
            metadata: Optional metadata
            
        Returns:
            Lineage tracking identifier
        """
        pass
    
    @abstractmethod
    async def get_data_lineage(
        self, 
        data_identifier: str
    ) -> Dict[str, Any]:
        """Get lineage information for data.
        
        Args:
            data_identifier: Data identifier
            
        Returns:
            Lineage information
        """
        pass
    
    @abstractmethod
    async def get_upstream_dependencies(
        self, 
        data_identifier: str
    ) -> List[str]:
        """Get upstream dependencies for data.
        
        Args:
            data_identifier: Data identifier
            
        Returns:
            List of upstream data identifiers
        """
        pass
    
    @abstractmethod
    async def get_downstream_dependencies(
        self, 
        data_identifier: str
    ) -> List[str]:
        """Get downstream dependencies for data.
        
        Args:
            data_identifier: Data identifier
            
        Returns:
            List of downstream data identifiers
        """
        pass
    
    @abstractmethod
    async def analyze_impact(
        self, 
        data_identifier: str, 
        change_description: str
    ) -> Dict[str, Any]:
        """Analyze impact of changes to data.
        
        Args:
            data_identifier: Data identifier
            change_description: Description of planned changes
            
        Returns:
            Impact analysis results
        """
        pass