"""Repository interface for quality assessment entities and reports."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional, List, Dict
from uuid import UUID

from packages.core.domain.abstractions.repository_interface import RepositoryInterface
from ..entities.data_profile import QualityAssessment
from ..value_objects.quality_metrics import (
    QualityReport, QualityRule, QualityViolation, QualityConfiguration,
    QualityDimension, SeverityLevel, QualityRuleType
)


class QualityAssessmentRepository(RepositoryInterface[QualityAssessment], ABC):
    """Repository interface for quality assessment persistence operations."""

    @abstractmethod
    async def find_by_dataset_id(self, dataset_id: str) -> List[QualityAssessment]:
        """Find quality assessments by dataset ID.
        
        Args:
            dataset_id: Dataset ID to search for
            
        Returns:
            List of quality assessments for the dataset
        """
        pass

    @abstractmethod
    async def find_by_score_range(
        self, min_score: float, max_score: float
    ) -> List[QualityAssessment]:
        """Find quality assessments by score range.
        
        Args:
            min_score: Minimum quality score
            max_score: Maximum quality score
            
        Returns:
            List of quality assessments within the score range
        """
        pass

    @abstractmethod
    async def find_assessments_with_critical_issues(self) -> List[QualityAssessment]:
        """Find assessments with critical quality issues.
        
        Returns:
            List of assessments with critical issues
        """
        pass

    @abstractmethod
    async def find_by_date_range(
        self, start_date: datetime, end_date: datetime
    ) -> List[QualityAssessment]:
        """Find assessments within a date range.
        
        Args:
            start_date: Start of date range
            end_date: End of date range
            
        Returns:
            List of assessments within the date range
        """
        pass

    @abstractmethod
    async def get_quality_trends(
        self, dataset_id: str, days: int = 30
    ) -> List[Dict[str, Any]]:
        """Get quality trends for a dataset over time.
        
        Args:
            dataset_id: Dataset ID
            days: Number of days to analyze
            
        Returns:
            List of quality trend data points
        """
        pass

    @abstractmethod
    async def get_dimension_statistics(
        self, dimension: QualityDimension
    ) -> Dict[str, Any]:
        """Get statistics for a specific quality dimension.
        
        Args:
            dimension: Quality dimension to analyze
            
        Returns:
            Dictionary containing dimension statistics
        """
        pass

    @abstractmethod
    async def find_similar_quality_profiles(
        self, assessment_id: UUID, similarity_threshold: float = 0.8
    ) -> List[tuple[QualityAssessment, float]]:
        """Find assessments with similar quality profiles.
        
        Args:
            assessment_id: Reference assessment ID
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of tuples (similar_assessment, similarity_score)
        """
        pass

    @abstractmethod
    async def get_quality_benchmarks(self) -> Dict[QualityDimension, float]:
        """Get quality benchmarks across all assessments.
        
        Returns:
            Dictionary mapping dimensions to benchmark scores
        """
        pass

    @abstractmethod
    async def find_assessments_below_threshold(
        self, dimension: QualityDimension, threshold: float
    ) -> List[QualityAssessment]:
        """Find assessments below quality threshold.
        
        Args:
            dimension: Quality dimension to check
            threshold: Minimum acceptable score
            
        Returns:
            List of assessments below threshold
        """
        pass

    @abstractmethod
    async def get_global_quality_statistics(self) -> Dict[str, Any]:
        """Get global quality statistics across all assessments.
        
        Returns:
            Dictionary containing global quality statistics
        """
        pass


class QualityReportRepository(ABC):
    """Repository interface for quality report persistence operations."""

    @abstractmethod
    async def save(self, report: QualityReport) -> QualityReport:
        """Save quality report.
        
        Args:
            report: Quality report to save
            
        Returns:
            Saved quality report
        """
        pass

    @abstractmethod
    async def find_by_id(self, report_id: str) -> Optional[QualityReport]:
        """Find quality report by ID.
        
        Args:
            report_id: Report ID to search for
            
        Returns:
            QualityReport if found, None otherwise
        """
        pass

    @abstractmethod
    async def find_by_dataset_id(self, dataset_id: str) -> List[QualityReport]:
        """Find quality reports by dataset ID.
        
        Args:
            dataset_id: Dataset ID to search for
            
        Returns:
            List of quality reports for the dataset
        """
        pass

    @abstractmethod
    async def find_latest_report(self, dataset_id: str) -> Optional[QualityReport]:
        """Find latest quality report for a dataset.
        
        Args:
            dataset_id: Dataset ID to search for
            
        Returns:
            Latest QualityReport if found, None otherwise
        """
        pass

    @abstractmethod
    async def find_reports_with_violations(
        self, severity: Optional[SeverityLevel] = None
    ) -> List[QualityReport]:
        """Find reports with quality violations.
        
        Args:
            severity: Optional severity filter
            
        Returns:
            List of reports with violations
        """
        pass

    @abstractmethod
    async def get_report_comparison(
        self, report_id_1: str, report_id_2: str
    ) -> Dict[str, Any]:
        """Compare two quality reports.
        
        Args:
            report_id_1: First report ID
            report_id_2: Second report ID
            
        Returns:
            Dictionary containing comparison results
        """
        pass

    @abstractmethod
    async def get_quality_evolution(
        self, dataset_id: str, days: int = 30
    ) -> List[Dict[str, Any]]:
        """Get quality evolution over time for a dataset.
        
        Args:
            dataset_id: Dataset ID
            days: Number of days to analyze
            
        Returns:
            List of quality evolution data points
        """
        pass

    @abstractmethod
    async def archive_old_reports(self, older_than_days: int = 90) -> int:
        """Archive old quality reports.
        
        Args:
            older_than_days: Archive reports older than this many days
            
        Returns:
            Number of reports archived
        """
        pass


class QualityRuleRepository(ABC):
    """Repository interface for quality rule persistence operations."""

    @abstractmethod
    async def save(self, rule: QualityRule) -> QualityRule:
        """Save quality rule.
        
        Args:
            rule: Quality rule to save
            
        Returns:
            Saved quality rule
        """
        pass

    @abstractmethod
    async def find_by_id(self, rule_id: str) -> Optional[QualityRule]:
        """Find quality rule by ID.
        
        Args:
            rule_id: Rule ID to search for
            
        Returns:
            QualityRule if found, None otherwise
        """
        pass

    @abstractmethod
    async def find_all_active(self) -> List[QualityRule]:
        """Find all active quality rules.
        
        Returns:
            List of active quality rules
        """
        pass

    @abstractmethod
    async def find_by_type(self, rule_type: QualityRuleType) -> List[QualityRule]:
        """Find quality rules by type.
        
        Args:
            rule_type: Rule type to search for
            
        Returns:
            List of rules of the specified type
        """
        pass

    @abstractmethod
    async def find_by_dimension(self, dimension: QualityDimension) -> List[QualityRule]:
        """Find quality rules by dimension.
        
        Args:
            dimension: Quality dimension to search for
            
        Returns:
            List of rules for the dimension
        """
        pass

    @abstractmethod
    async def find_by_column(self, column_name: str) -> List[QualityRule]:
        """Find quality rules targeting specific column.
        
        Args:
            column_name: Column name to search for
            
        Returns:
            List of rules targeting the column
        """
        pass

    @abstractmethod
    async def find_by_severity(self, severity: SeverityLevel) -> List[QualityRule]:
        """Find quality rules by severity level.
        
        Args:
            severity: Severity level to search for
            
        Returns:
            List of rules with the specified severity
        """
        pass

    @abstractmethod
    async def update_rule_status(self, rule_id: str, is_active: bool) -> bool:
        """Update rule active status.
        
        Args:
            rule_id: Rule ID to update
            is_active: New active status
            
        Returns:
            True if update was successful, False otherwise
        """
        pass

    @abstractmethod
    async def delete_rule(self, rule_id: str) -> bool:
        """Delete quality rule.
        
        Args:
            rule_id: Rule ID to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        pass

    @abstractmethod
    async def get_rule_usage_statistics(self) -> Dict[str, Any]:
        """Get usage statistics for quality rules.
        
        Returns:
            Dictionary containing rule usage statistics
        """
        pass

    @abstractmethod
    async def validate_rule_expression(self, rule_id: str) -> Dict[str, Any]:
        """Validate rule expression syntax.
        
        Args:
            rule_id: Rule ID to validate
            
        Returns:
            Dictionary containing validation results
        """
        pass


class QualityViolationRepository(ABC):
    """Repository interface for quality violation persistence operations."""

    @abstractmethod
    async def save(self, violation: QualityViolation) -> QualityViolation:
        """Save quality violation.
        
        Args:
            violation: Quality violation to save
            
        Returns:
            Saved quality violation
        """
        pass

    @abstractmethod
    async def find_by_report_id(self, report_id: str) -> List[QualityViolation]:
        """Find violations by report ID.
        
        Args:
            report_id: Report ID to search for
            
        Returns:
            List of violations for the report
        """
        pass

    @abstractmethod
    async def find_by_rule_id(self, rule_id: str) -> List[QualityViolation]:
        """Find violations by rule ID.
        
        Args:
            rule_id: Rule ID to search for
            
        Returns:
            List of violations for the rule
        """
        pass

    @abstractmethod
    async def find_by_severity(self, severity: SeverityLevel) -> List[QualityViolation]:
        """Find violations by severity level.
        
        Args:
            severity: Severity level to search for
            
        Returns:
            List of violations with the specified severity
        """
        pass

    @abstractmethod
    async def find_by_column(self, column_name: str) -> List[QualityViolation]:
        """Find violations by column name.
        
        Args:
            column_name: Column name to search for
            
        Returns:
            List of violations for the column
        """
        pass

    @abstractmethod
    async def find_critical_violations(self) -> List[QualityViolation]:
        """Find all critical violations.
        
        Returns:
            List of critical violations
        """
        pass

    @abstractmethod
    async def get_violation_trends(
        self, days: int = 30
    ) -> Dict[str, List[int]]:
        """Get violation trends over time.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dictionary with violation trend data
        """
        pass

    @abstractmethod
    async def get_violation_statistics_by_rule(self) -> Dict[str, Dict[str, Any]]:
        """Get violation statistics grouped by rule.
        
        Returns:
            Dictionary mapping rule IDs to violation statistics
        """
        pass

    @abstractmethod
    async def mark_violation_resolved(
        self, violation_id: str, resolution_notes: str
    ) -> bool:
        """Mark violation as resolved.
        
        Args:
            violation_id: Violation ID to resolve
            resolution_notes: Notes about the resolution
            
        Returns:
            True if resolution was successful, False otherwise
        """
        pass

    @abstractmethod
    async def bulk_resolve_violations(
        self, violation_ids: List[str], resolution_notes: str
    ) -> int:
        """Bulk resolve multiple violations.
        
        Args:
            violation_ids: List of violation IDs to resolve
            resolution_notes: Notes about the resolution
            
        Returns:
            Number of violations resolved
        """
        pass


class QualityConfigurationRepository(ABC):
    """Repository interface for quality configuration persistence operations."""

    @abstractmethod
    async def save(self, config: QualityConfiguration) -> QualityConfiguration:
        """Save quality configuration.
        
        Args:
            config: Quality configuration to save
            
        Returns:
            Saved quality configuration
        """
        pass

    @abstractmethod
    async def find_by_id(self, config_id: str) -> Optional[QualityConfiguration]:
        """Find quality configuration by ID.
        
        Args:
            config_id: Configuration ID to search for
            
        Returns:
            QualityConfiguration if found, None otherwise
        """
        pass

    @abstractmethod
    async def find_active_configurations(self) -> List[QualityConfiguration]:
        """Find all active quality configurations.
        
        Returns:
            List of active quality configurations
        """
        pass

    @abstractmethod
    async def find_by_name(self, name: str) -> Optional[QualityConfiguration]:
        """Find quality configuration by name.
        
        Args:
            name: Configuration name to search for
            
        Returns:
            QualityConfiguration if found, None otherwise
        """
        pass

    @abstractmethod
    async def get_default_configuration(self) -> Optional[QualityConfiguration]:
        """Get default quality configuration.
        
        Returns:
            Default QualityConfiguration if exists, None otherwise
        """
        pass

    @abstractmethod
    async def clone_configuration(
        self, source_config_id: str, new_name: str
    ) -> Optional[QualityConfiguration]:
        """Clone existing configuration with new name.
        
        Args:
            source_config_id: Source configuration ID
            new_name: New configuration name
            
        Returns:
            Cloned QualityConfiguration if successful, None otherwise
        """
        pass

    @abstractmethod
    async def update_configuration_status(
        self, config_id: str, is_active: bool
    ) -> bool:
        """Update configuration active status.
        
        Args:
            config_id: Configuration ID to update
            is_active: New active status
            
        Returns:
            True if update was successful, False otherwise
        """
        pass

    @abstractmethod
    async def validate_configuration(
        self, config_id: str
    ) -> Dict[str, Any]:
        """Validate quality configuration.
        
        Args:
            config_id: Configuration ID to validate
            
        Returns:
            Dictionary containing validation results
        """
        pass