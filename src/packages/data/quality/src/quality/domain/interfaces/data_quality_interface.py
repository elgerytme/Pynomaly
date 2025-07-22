"""
Data Quality Interface for defining contracts for quality operations.

This module provides the main interface for data quality services and operations.
It defines the contract that must be implemented by any service that provides
data quality functionality within the quality domain.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from ..entities.quality_profile import DataQualityProfile
from ..entities.quality_scores import QualityScores
from ..entities.validation_rule import ValidationResult
from ..entities.quality_issue import QualityIssue


class QualityOperationType(Enum):
    """Types of data quality operations."""
    VALIDATION = "validation"
    PROFILING = "profiling"
    CLEANSING = "cleansing"
    MONITORING = "monitoring"
    ASSESSMENT = "assessment"
    REMEDIATION = "remediation"


class ReportLevel(Enum):
    """Report detail levels."""
    SUMMARY = "summary"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"


@dataclass
class QualityMetrics:
    """Quality metrics container."""
    accuracy_score: float = 0.0
    completeness_score: float = 0.0
    consistency_score: float = 0.0
    validity_score: float = 0.0
    uniqueness_score: float = 0.0
    timeliness_score: float = 0.0
    overall_score: float = 0.0
    
    # Additional metrics
    record_count: int = 0
    error_count: int = 0
    warning_count: int = 0
    
    # Timestamps
    calculated_at: datetime = field(default_factory=datetime.utcnow)
    data_timestamp: Optional[datetime] = None


@dataclass
class QualityReport:
    """Quality report containing assessment results and recommendations."""
    report_id: str
    operation_type: QualityOperationType
    dataset_id: str
    
    # Report metadata
    generated_at: datetime = field(default_factory=datetime.utcnow)
    generated_by: str = "system"
    report_level: ReportLevel = ReportLevel.SUMMARY
    
    # Quality assessment results
    metrics: QualityMetrics = field(default_factory=QualityMetrics)
    validation_results: List[ValidationResult] = field(default_factory=list)
    quality_issues: List[QualityIssue] = field(default_factory=list)
    
    # Analysis and recommendations
    summary: str = ""
    findings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Configuration and context
    configuration_used: Dict[str, Any] = field(default_factory=dict)
    execution_context: Dict[str, Any] = field(default_factory=dict)
    
    # Status and metadata
    status: str = "completed"
    error_message: Optional[str] = None
    processing_time_seconds: float = 0.0
    
    @property
    def has_issues(self) -> bool:
        """Check if report contains quality issues."""
        return len(self.quality_issues) > 0
    
    @property
    def issue_count_by_severity(self) -> Dict[str, int]:
        """Get count of issues by severity."""
        counts = {}
        for issue in self.quality_issues:
            severity = issue.severity.value if hasattr(issue.severity, 'value') else str(issue.severity)
            counts[severity] = counts.get(severity, 0) + 1
        return counts


class DataQualityInterface(ABC):
    """
    Abstract interface for data quality operations.
    
    This interface defines the contract that must be implemented by any service
    that provides data quality functionality. It follows domain-driven design
    principles and provides a clean abstraction layer.
    """
    
    @abstractmethod
    async def validate_data(
        self,
        dataset_id: str,
        validation_rules: List[Dict[str, Any]],
        configuration: Optional[Dict[str, Any]] = None
    ) -> QualityReport:
        """
        Validate data against specified rules.
        
        Args:
            dataset_id: Identifier of the dataset to validate
            validation_rules: List of validation rules to apply
            configuration: Optional configuration parameters
            
        Returns:
            QualityReport containing validation results
        """
        pass
    
    @abstractmethod
    async def profile_data(
        self,
        dataset_id: str,
        configuration: Optional[Dict[str, Any]] = None
    ) -> DataQualityProfile:
        """
        Generate a comprehensive data quality profile.
        
        Args:
            dataset_id: Identifier of the dataset to profile
            configuration: Optional configuration parameters
            
        Returns:
            DataQualityProfile containing profiling results
        """
        pass
    
    @abstractmethod
    async def assess_quality(
        self,
        dataset_id: str,
        assessment_criteria: Optional[Dict[str, Any]] = None,
        configuration: Optional[Dict[str, Any]] = None
    ) -> QualityReport:
        """
        Assess overall data quality.
        
        Args:
            dataset_id: Identifier of the dataset to assess
            assessment_criteria: Criteria for quality assessment
            configuration: Optional configuration parameters
            
        Returns:
            QualityReport containing quality assessment results
        """
        pass
    
    @abstractmethod
    async def calculate_quality_scores(
        self,
        dataset_id: str,
        metrics_configuration: Optional[Dict[str, Any]] = None
    ) -> QualityScores:
        """
        Calculate comprehensive quality scores.
        
        Args:
            dataset_id: Identifier of the dataset
            metrics_configuration: Configuration for metrics calculation
            
        Returns:
            QualityScores object with calculated scores
        """
        pass
    
    @abstractmethod
    async def detect_quality_issues(
        self,
        dataset_id: str,
        detection_rules: Optional[List[Dict[str, Any]]] = None,
        configuration: Optional[Dict[str, Any]] = None
    ) -> List[QualityIssue]:
        """
        Detect quality issues in the dataset.
        
        Args:
            dataset_id: Identifier of the dataset
            detection_rules: Rules for issue detection
            configuration: Optional configuration parameters
            
        Returns:
            List of detected quality issues
        """
        pass
    
    @abstractmethod
    async def monitor_quality(
        self,
        dataset_id: str,
        monitoring_configuration: Dict[str, Any]
    ) -> QualityReport:
        """
        Monitor data quality over time.
        
        Args:
            dataset_id: Identifier of the dataset to monitor
            monitoring_configuration: Configuration for monitoring
            
        Returns:
            QualityReport containing monitoring results
        """
        pass
    
    @abstractmethod
    async def get_quality_metrics(
        self,
        dataset_id: str,
        metric_types: Optional[List[str]] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None
    ) -> QualityMetrics:
        """
        Get quality metrics for a dataset.
        
        Args:
            dataset_id: Identifier of the dataset
            metric_types: Specific metrics to retrieve
            time_range: Time range for metrics (start, end)
            
        Returns:
            QualityMetrics containing the requested metrics
        """
        pass
    
    @abstractmethod
    async def generate_quality_report(
        self,
        dataset_id: str,
        report_type: QualityOperationType,
        report_level: ReportLevel = ReportLevel.SUMMARY,
        configuration: Optional[Dict[str, Any]] = None
    ) -> QualityReport:
        """
        Generate a comprehensive quality report.
        
        Args:
            dataset_id: Identifier of the dataset
            report_type: Type of quality report to generate
            report_level: Detail level of the report
            configuration: Optional configuration parameters
            
        Returns:
            QualityReport containing the generated report
        """
        pass
    
    @abstractmethod
    async def cleanup_resources(self) -> None:
        """
        Clean up any resources used by the implementation.
        
        This method should be called when the service is being shut down
        to ensure proper resource cleanup.
        """
        pass
    
    # Optional helper methods that implementations can override
    
    def get_supported_operations(self) -> List[QualityOperationType]:
        """
        Get list of supported quality operations.
        
        Returns:
            List of supported operation types
        """
        return [op for op in QualityOperationType]
    
    def get_configuration_schema(self) -> Dict[str, Any]:
        """
        Get configuration schema for the implementation.
        
        Returns:
            Dictionary describing the configuration schema
        """
        return {
            "type": "object",
            "properties": {
                "max_concurrent_operations": {"type": "integer", "default": 5},
                "timeout_seconds": {"type": "integer", "default": 300},
                "enable_caching": {"type": "boolean", "default": True},
                "cache_ttl_seconds": {"type": "integer", "default": 3600},
                "enable_detailed_logging": {"type": "boolean", "default": False}
            }
        }
    
    def validate_configuration(self, configuration: Dict[str, Any]) -> bool:
        """
        Validate configuration parameters.
        
        Args:
            configuration: Configuration dictionary to validate
            
        Returns:
            True if configuration is valid, False otherwise
        """
        # Basic validation - implementations can override
        required_types = {
            "max_concurrent_operations": int,
            "timeout_seconds": int,
            "enable_caching": bool,
            "cache_ttl_seconds": int,
            "enable_detailed_logging": bool
        }
        
        for key, value in configuration.items():
            if key in required_types:
                if not isinstance(value, required_types[key]):
                    return False
        
        return True