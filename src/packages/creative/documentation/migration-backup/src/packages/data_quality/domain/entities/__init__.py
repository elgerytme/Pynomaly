"""Data Quality Domain Entities.

This module contains the core domain entities for the data quality validation system,
including quality profiles, validation rules, results, and issues.
"""

from .quality_profile import (
    DataQualityProfile, QualityJob, QualityJobConfig, JobMetrics,
    ProfileId, DatasetId, JobId, ProfileVersion
)
from .validation_rule import (
    QualityRule, ValidationLogic, ValidationResult, ValidationError,
    RuleId, ValidationId, RuleType, LogicType, ValidationStatus,
    Severity, QualityCategory, SuccessCriteria
)
from .quality_issue import (
    QualityIssue, RemediationSuggestion, BusinessImpact,
    IssueId, SuggestionId, QualityIssueType, IssueStatus,
    ImpactLevel, ComplianceRisk, CustomerImpact, OperationalImpact,
    RemediationAction, EffortEstimate, Priority
)
from .quality_scores import (
    QualityScores, QualityTrends, ScoringMethod, MonetaryAmount
)

__all__ = [
    # Quality Profile entities
    'DataQualityProfile',
    'QualityJob',
    'QualityJobConfig',
    'JobMetrics',
    'ProfileId',
    'DatasetId',
    'JobId',
    'ProfileVersion',
    
    # Validation Rule entities
    'QualityRule',
    'ValidationLogic',
    'ValidationResult',
    'ValidationError',
    'RuleId',
    'ValidationId',
    'RuleType',
    'LogicType',
    'ValidationStatus',
    'Severity',
    'QualityCategory',
    'SuccessCriteria',
    
    # Quality Issue entities
    'QualityIssue',
    'RemediationSuggestion',
    'BusinessImpact',
    'IssueId',
    'SuggestionId',
    'QualityIssueType',
    'IssueStatus',
    'ImpactLevel',
    'ComplianceRisk',
    'CustomerImpact',
    'OperationalImpact',
    'RemediationAction',
    'EffortEstimate',
    'Priority',
    
    # Quality Scores entities
    'QualityScores',
    'QualityTrends',
    'ScoringMethod',
    'MonetaryAmount'
]