"""Quality Assessment Service.

Service for comprehensive data quality assessment including 6-dimensional scoring,
weighted scoring, issue detection, and quality trend analysis.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging
from collections import defaultdict
from enum import Enum

from ...domain.entities.quality_profile import DataQualityProfile, DatasetId, ProfileId, ProfileVersion
from ...domain.entities.quality_scores import QualityScores, QualityTrends, QualityTrendPoint, ScoringMethod
from ...domain.entities.quality_issue import (
    QualityIssue, BusinessImpact, RemediationSuggestion, 
    IssueId, SuggestionId, QualityIssueType, IssueStatus,
    ImpactLevel, ComplianceRisk, CustomerImpact, OperationalImpact,
    RemediationAction, EffortEstimate, Priority
)
from ...domain.entities.validation_rule import ValidationResult, ValidationStatus, Severity
from .validation_engine import ValidationEngine

logger = logging.getLogger(__name__)


class QualityDimension(Enum):
    """Quality dimensions for assessment."""
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    VALIDITY = "validity"
    UNIQUENESS = "uniqueness"
    TIMELINESS = "timeliness"


@dataclass(frozen=True)
class QualityAssessmentConfig:
    """Configuration for quality assessment."""
    scoring_method: ScoringMethod = ScoringMethod.WEIGHTED_AVERAGE
    dimension_weights: Dict[str, float] = field(default_factory=lambda: {
        'completeness': 0.20,
        'accuracy': 0.25,
        'consistency': 0.20,
        'validity': 0.20,
        'uniqueness': 0.10,
        'timeliness': 0.05
    })
    
    # Thresholds for issue detection
    critical_threshold: float = 0.5
    high_threshold: float = 0.7
    medium_threshold: float = 0.8
    low_threshold: float = 0.9
    
    # Analysis settings
    enable_trend_analysis: bool = True
    trend_period_days: int = 30
    enable_business_impact_analysis: bool = True
    enable_remediation_suggestions: bool = True
    
    # Performance settings
    sample_size_for_analysis: int = 50000
    max_issues_per_type: int = 1000
    
    def __post_init__(self):
        """Validate configuration."""
        # Validate weights sum to 1.0
        weight_sum = sum(self.dimension_weights.values())
        if abs(weight_sum - 1.0) > 0.001:
            raise ValueError(f"Dimension weights must sum to 1.0, got {weight_sum}")
        
        # Validate thresholds
        thresholds = [self.critical_threshold, self.high_threshold, self.medium_threshold, self.low_threshold]
        if not all(0.0 <= t <= 1.0 for t in thresholds):
            raise ValueError("All thresholds must be between 0 and 1")
        
        if not all(thresholds[i] <= thresholds[i+1] for i in range(len(thresholds)-1)):
            raise ValueError("Thresholds must be in ascending order")


@dataclass(frozen=True)
class DimensionAssessment:
    """Assessment for a single quality dimension."""
    dimension: QualityDimension
    score: float
    issue_count: int
    details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    
    def get_quality_level(self) -> str:
        """Get quality level based on score."""
        if self.score >= 0.9:
            return "excellent"
        elif self.score >= 0.8:
            return "good"
        elif self.score >= 0.7:
            return "fair"
        elif self.score >= 0.5:
            return "poor"
        else:
            return "critical"


class QualityAssessmentService:
    """Service for comprehensive quality assessment."""
    
    def __init__(self, config: QualityAssessmentConfig = None):
        """Initialize quality assessment service."""
        self.config = config or QualityAssessmentConfig()
        self.validation_engine = ValidationEngine()
        
        # Assessment statistics
        self._assessment_stats = {
            'total_assessments': 0,
            'average_overall_score': 0.0,
            'total_issues_detected': 0,
            'most_common_issue_type': None,
            'assessment_duration_seconds': 0.0
        }
    
    def assess_dataset_quality(self, 
                              df: pd.DataFrame,
                              validation_results: List[ValidationResult] = None,
                              dataset_id: DatasetId = None,
                              previous_profile: DataQualityProfile = None) -> DataQualityProfile:
        """Perform comprehensive quality assessment on dataset."""
        start_time = datetime.now()
        
        try:
            # Create dataset ID if not provided
            if not dataset_id:
                dataset_id = DatasetId(f"dataset_{int(datetime.now().timestamp())}")
            
            # Apply sampling if dataset is large
            if len(df) > self.config.sample_size_for_analysis:
                df_sample = df.sample(n=self.config.sample_size_for_analysis, random_state=42)
                logger.info(f"Applied sampling for assessment: {len(df_sample)} rows")
            else:
                df_sample = df
            
            # Assess individual quality dimensions
            dimension_assessments = self._assess_all_dimensions(df_sample)
            
            # Calculate overall quality scores
            quality_scores = self._calculate_quality_scores(dimension_assessments)
            
            # Detect quality issues
            quality_issues = self._detect_quality_issues(df_sample, dimension_assessments, validation_results)
            
            # Generate remediation suggestions
            remediation_suggestions = self._generate_remediation_suggestions(quality_issues) if self.config.enable_remediation_suggestions else []
            
            # Analyze quality trends
            quality_trends = self._analyze_quality_trends(quality_scores, previous_profile) if self.config.enable_trend_analysis else QualityTrends([])
            
            # Create quality profile
            profile = DataQualityProfile(
                profile_id=ProfileId(),
                dataset_id=dataset_id,
                quality_scores=quality_scores,
                validation_results=validation_results or [],
                quality_issues=quality_issues,
                remediation_suggestions=remediation_suggestions,
                quality_trends=quality_trends,
                created_at=start_time,
                last_assessed=datetime.now(),
                version=ProfileVersion(),
                record_count=len(df),
                column_count=len(df.columns),
                data_size_bytes=df.memory_usage(deep=True).sum()
            )
            
            # Update statistics
            self._update_assessment_stats(profile, start_time)
            
            logger.info(f"Quality assessment completed for dataset {dataset_id}")
            return profile
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {str(e)}")
            raise
    
    def compare_quality_profiles(self, 
                               profile1: DataQualityProfile,
                               profile2: DataQualityProfile) -> Dict[str, Any]:
        """Compare two quality profiles."""
        comparison = {
            'profile1_id': str(profile1.profile_id),
            'profile2_id': str(profile2.profile_id),
            'comparison_date': datetime.now().isoformat(),
            'score_comparison': profile1.quality_scores.compare_with(profile2.quality_scores),
            'issue_comparison': self._compare_issues(profile1.quality_issues, profile2.quality_issues),
            'trend_analysis': self._compare_trends(profile1, profile2),
            'recommendations': self._generate_comparison_recommendations(profile1, profile2)
        }
        
        return comparison

    def get_assessment_statistics(self) -> Dict[str, Any]:
        """Get assessment service statistics."""
        return self._assessment_stats.copy()