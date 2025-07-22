"""Advanced Quality Metrics and Scoring Service.

Service for calculating comprehensive quality metrics, advanced scoring algorithms,
and providing detailed quality analytics with multi-dimensional assessments.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging
from collections import defaultdict, Counter
from enum import Enum
import math
from statistics import mean, median, stdev, harmonic_mean

from ...domain.entities.quality_profile import DataQualityProfile, DatasetId
from ...domain.entities.quality_scores import QualityScores
# TODO: QualityIssue entity needs to be created or use QualityIssue instead
from ...domain.entities.quality_issue import QualityIssue, ImpactLevel
from ...domain.entities.validation_rule import QualityRule, QualityCategory
from .quality_assessment_service import QualityAssessmentService

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of quality metrics."""
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    VALIDITY = "validity"
    UNIQUENESS = "uniqueness"
    TIMELINESS = "timeliness"
    CONFORMITY = "conformity"
    INTEGRITY = "integrity"
    PRECISION = "precision"
    RELEVANCE = "relevance"


class ScoringMethod(Enum):
    """Scoring methods for quality assessment."""
    WEIGHTED_AVERAGE = "weighted_average"
    GEOMETRIC_MEAN = "geometric_mean"
    HARMONIC_MEAN = "harmonic_mean"
    MIN_MAX_NORMALIZATION = "min_max_normalization"
    Z_SCORE_NORMALIZATION = "z_score_normalization"
    PERCENTILE_RANKING = "percentile_ranking"
    FUZZY_LOGIC = "fuzzy_logic"
    MULTI_CRITERIA = "multi_criteria"


class QualityTier(Enum):
    """Quality tiers for classification."""
    EXCELLENT = "excellent"      # 95-100%
    GOOD = "good"               # 80-94%
    FAIR = "fair"               # 60-79%
    POOR = "poor"               # 40-59%
    CRITICAL = "critical"       # 0-39%


@dataclass(frozen=True)
class MetricWeight:
    """Weight configuration for quality metrics."""
    metric_type: MetricType
    weight: float
    critical_threshold: float = 0.5
    target_value: float = 1.0
    
    def __post_init__(self):
        if not 0 <= self.weight <= 1:
            raise ValueError("Weight must be between 0 and 1")
        if not 0 <= self.critical_threshold <= 1:
            raise ValueError("Critical threshold must be between 0 and 1")


@dataclass(frozen=True)
class AdvancedMetricsConfig:
    """Configuration for advanced quality metrics."""
    # Default metric weights
    metric_weights: List[MetricWeight] = field(default_factory=lambda: [
        MetricWeight(MetricType.COMPLETENESS, 0.2, 0.8),
        MetricWeight(MetricType.ACCURACY, 0.25, 0.7),
        MetricWeight(MetricType.CONSISTENCY, 0.15, 0.6),
        MetricWeight(MetricType.VALIDITY, 0.2, 0.8),
        MetricWeight(MetricType.UNIQUENESS, 0.1, 0.9),
        MetricWeight(MetricType.TIMELINESS, 0.1, 0.5)
    ])
    
    # Scoring configuration
    default_scoring_method: ScoringMethod = ScoringMethod.WEIGHTED_AVERAGE
    enable_fuzzy_logic: bool = True
    enable_multi_criteria: bool = True
    
    # Normalization settings
    enable_z_score_normalization: bool = True
    outlier_threshold: float = 3.0
    
    # Temporal analysis
    enable_temporal_scoring: bool = True
    temporal_window_days: int = 30
    temporal_decay_factor: float = 0.95
    
    # Benchmarking
    enable_benchmarking: bool = True
    benchmark_percentiles: List[float] = field(default_factory=lambda: [25, 50, 75, 90, 95])
    
    # Advanced features
    enable_domain_specific_scoring: bool = True
    enable_business_impact_weighting: bool = True
    confidence_interval_alpha: float = 0.05


@dataclass
class QualityMetric:
    """Individual quality metric result."""
    metric_type: MetricType
    value: float
    confidence: float
    weight: float
    normalized_value: float
    benchmark_percentile: Optional[float] = None
    
    # Detailed breakdown
    sub_metrics: Dict[str, float] = field(default_factory=dict)
    contributing_factors: Dict[str, float] = field(default_factory=dict)
    
    # Temporal information
    trend_direction: Optional[str] = None
    trend_significance: Optional[float] = None
    historical_average: Optional[float] = None
    
    # Metadata
    calculation_method: str = "default"
    calculation_timestamp: datetime = field(default_factory=datetime.now)
    
    def get_quality_tier(self) -> QualityTier:
        """Get quality tier based on value."""
        if self.value >= 0.95:
            return QualityTier.EXCELLENT
        elif self.value >= 0.8:
            return QualityTier.GOOD
        elif self.value >= 0.6:
            return QualityTier.FAIR
        elif self.value >= 0.4:
            return QualityTier.POOR
        else:
            return QualityTier.CRITICAL
    
    def is_critical(self) -> bool:
        """Check if metric is in critical state."""
        return self.value < 0.4
    
    def get_metric_summary(self) -> Dict[str, Any]:
        """Get metric summary."""
        return {
            'metric_type': self.metric_type.value,
            'value': round(self.value, 4),
            'normalized_value': round(self.normalized_value, 4),
            'confidence': round(self.confidence, 3),
            'weight': self.weight,
            'quality_tier': self.get_quality_tier().value,
            'is_critical': self.is_critical(),
            'benchmark_percentile': self.benchmark_percentile,
            'trend_direction': self.trend_direction,
            'sub_metrics': {k: round(v, 4) for k, v in self.sub_metrics.items()},
            'calculation_method': self.calculation_method,
            'calculation_timestamp': self.calculation_timestamp.isoformat()
        }


@dataclass
class AdvancedQualityScore:
    """Advanced quality score with detailed breakdown."""
    dataset_id: str
    overall_score: float
    normalized_score: float
    quality_tier: QualityTier
    confidence_interval: Tuple[float, float]
    
    # Detailed metrics
    metrics: Dict[MetricType, QualityMetric]
    
    # Scoring information
    scoring_method: ScoringMethod
    weights_used: Dict[MetricType, float]
    
    # Comparative analysis
    benchmark_scores: Dict[str, float] = field(default_factory=dict)
    percentile_ranking: Optional[float] = None
    
    # Temporal analysis
    trend_analysis: Dict[str, Any] = field(default_factory=dict)
    historical_comparison: Dict[str, Any] = field(default_factory=dict)
    
    # Business impact
    business_impact_score: Optional[float] = None
    critical_issues: List[str] = field(default_factory=list)
    improvement_recommendations: List[str] = field(default_factory=list)
    
    # Metadata
    calculation_timestamp: datetime = field(default_factory=datetime.now)
    calculation_duration_ms: float = 0.0
    
    def get_critical_metrics(self) -> List[QualityMetric]:
        """Get metrics in critical state."""
        return [metric for metric in self.metrics.values() if metric.is_critical()]
    
    def get_top_improvement_areas(self, n: int = 3) -> List[Tuple[MetricType, float]]:
        """Get top areas for improvement."""
        sorted_metrics = sorted(
            self.metrics.items(),
            key=lambda x: x[1].value * x[1].weight,
            reverse=False
        )
        return [(metric_type, metric.value) for metric_type, metric in sorted_metrics[:n]]
    
    def get_quality_summary(self) -> Dict[str, Any]:
        """Get comprehensive quality summary."""
        return {
            'dataset_id': self.dataset_id,
            'overall_score': round(self.overall_score, 4),
            'normalized_score': round(self.normalized_score, 4),
            'quality_tier': self.quality_tier.value,
            'confidence_interval': [round(self.confidence_interval[0], 4), 
                                   round(self.confidence_interval[1], 4)],
            'scoring_method': self.scoring_method.value,
            'percentile_ranking': self.percentile_ranking,
            'business_impact_score': self.business_impact_score,
            'critical_issues_count': len(self.critical_issues),
            'critical_metrics': [m.metric_type.value for m in self.get_critical_metrics()],
            'top_improvement_areas': [
                {'metric': mt.value, 'score': round(score, 4)} 
                for mt, score in self.get_top_improvement_areas()
            ],
            'metrics_breakdown': {
                mt.value: metric.get_metric_summary() 
                for mt, metric in self.metrics.items()
            },
            'trend_analysis': self.trend_analysis,
            'calculation_timestamp': self.calculation_timestamp.isoformat(),
            'calculation_duration_ms': self.calculation_duration_ms
        }


class AdvancedQualityMetricsService:
    """Service for advanced quality metrics and scoring."""
    
    def __init__(self, config: AdvancedMetricsConfig = None):
        """Initialize advanced quality metrics service.
        
        Args:
            config: Service configuration
        """
        self.config = config or AdvancedMetricsConfig()
        self._historical_scores: Dict[str, List[AdvancedQualityScore]] = defaultdict(list)
        self._benchmark_data: Dict[str, List[float]] = defaultdict(list)
        
        # Create metric weights lookup
        self._metric_weights = {
            mw.metric_type: mw for mw in self.config.metric_weights
        }
        
        logger.info("Advanced Quality Metrics Service initialized")
    
    def calculate_advanced_score(self,
                                dataset_id: str,
                                data_profile: DataQualityProfile,
                                historical_data: Optional[List[Dict[str, Any]]] = None,
                                scoring_method: Optional[ScoringMethod] = None) -> AdvancedQualityScore:
        """Calculate advanced quality score with comprehensive analysis.
        
        Args:
            dataset_id: Dataset identifier
            data_profile: Data quality profile
            historical_data: Historical quality data for trends
            scoring_method: Specific scoring method to use
            
        Returns:
            Advanced quality score
        """
        start_time = datetime.now()
        
        # Calculate individual metrics
        metrics = self._calculate_detailed_metrics(data_profile, historical_data)
        
        # Select scoring method
        method = scoring_method or self.config.default_scoring_method
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(metrics, method)
        
        # Normalize score
        normalized_score = self._normalize_score(overall_score, metrics)
        
        # Determine quality tier
        quality_tier = self._determine_quality_tier(normalized_score)
        
        # Calculate confidence interval
        confidence_interval = self._calculate_confidence_interval(metrics)
        
        # Get weights used
        weights_used = {mt: self._metric_weights.get(mt, MetricWeight(mt, 0.0)).weight 
                       for mt in metrics.keys()}
        
        # Perform benchmarking if enabled
        benchmark_scores = {}
        percentile_ranking = None
        if self.config.enable_benchmarking:
            benchmark_scores, percentile_ranking = self._perform_benchmarking(
                dataset_id, overall_score
            )
        
        # Temporal analysis
        trend_analysis = {}
        historical_comparison = {}
        if self.config.enable_temporal_scoring and historical_data:
            trend_analysis = self._analyze_temporal_trends(historical_data)
            historical_comparison = self._compare_with_historical(overall_score, dataset_id)
        
        # Business impact analysis
        business_impact_score = None
        if self.config.enable_business_impact_weighting:
            business_impact_score = self._calculate_business_impact(metrics)
        
        # Generate critical issues and recommendations
        critical_issues = self._identify_critical_issues(metrics)
        recommendations = self._generate_improvement_recommendations(metrics)
        
        # Create advanced score
        calculation_duration = (datetime.now() - start_time).total_seconds() * 1000
        
        advanced_score = AdvancedQualityScore(
            dataset_id=dataset_id,
            overall_score=overall_score,
            normalized_score=normalized_score,
            quality_tier=quality_tier,
            confidence_interval=confidence_interval,
            metrics=metrics,
            scoring_method=method,
            weights_used=weights_used,
            benchmark_scores=benchmark_scores,
            percentile_ranking=percentile_ranking,
            trend_analysis=trend_analysis,
            historical_comparison=historical_comparison,
            business_impact_score=business_impact_score,
            critical_issues=critical_issues,
            improvement_recommendations=recommendations,
            calculation_duration_ms=calculation_duration
        )
        
        # Store for historical analysis
        self._historical_scores[dataset_id].append(advanced_score)
        
        # Update benchmark data
        if self.config.enable_benchmarking:
            self._benchmark_data[dataset_id].append(overall_score)
        
        logger.info(f"Calculated advanced score for {dataset_id}: {overall_score:.3f}")
        return advanced_score
    
    def _calculate_detailed_metrics(self,
                                  data_profile: DataQualityProfile,
                                  historical_data: Optional[List[Dict[str, Any]]] = None) -> Dict[MetricType, QualityMetric]:
        """Calculate detailed quality metrics."""
        metrics = {}
        
        # Completeness metric
        completeness = self._calculate_completeness_metric(data_profile)
        metrics[MetricType.COMPLETENESS] = completeness
        
        # Accuracy metric
        accuracy = self._calculate_accuracy_metric(data_profile)
        metrics[MetricType.ACCURACY] = accuracy
        
        # Consistency metric
        consistency = self._calculate_consistency_metric(data_profile)
        metrics[MetricType.CONSISTENCY] = consistency
        
        # Validity metric
        validity = self._calculate_validity_metric(data_profile)
        metrics[MetricType.VALIDITY] = validity
        
        # Uniqueness metric
        uniqueness = self._calculate_uniqueness_metric(data_profile)
        metrics[MetricType.UNIQUENESS] = uniqueness
        
        # Timeliness metric
        timeliness = self._calculate_timeliness_metric(data_profile)
        metrics[MetricType.TIMELINESS] = timeliness
        
        # Normalize metrics
        for metric_type, metric in metrics.items():
            metric.normalized_value = self._normalize_metric_value(metric.value)
        
        return metrics
    
    def _calculate_completeness_metric(self, data_profile: DataQualityProfile) -> QualityMetric:
        """Calculate completeness metric."""
        # Get completeness from profile
        base_completeness = data_profile.completeness_score
        
        # Calculate sub-metrics
        sub_metrics = {}
        contributing_factors = {}
        
        if hasattr(data_profile, 'column_profiles'):
            # Column-level completeness
            column_completeness = []
            for column_name, column_profile in data_profile.column_profiles.items():
                if hasattr(column_profile, 'completeness'):
                    column_completeness.append(column_profile.completeness)
                    sub_metrics[f'column_{column_name}_completeness'] = column_profile.completeness
            
            if column_completeness:
                contributing_factors['column_completeness_variance'] = np.var(column_completeness)
                contributing_factors['critical_columns_count'] = sum(1 for c in column_completeness if c < 0.8)
        
        # Row-level completeness
        if data_profile.row_count > 0:
            sub_metrics['row_completeness'] = base_completeness
            contributing_factors['total_rows'] = data_profile.row_count
        
        # Calculate confidence based on data size
        confidence = min(1.0, np.log10(data_profile.row_count) / 5) if data_profile.row_count > 0 else 0.5
        
        # Get weight
        weight = self._metric_weights.get(MetricType.COMPLETENESS, MetricWeight(MetricType.COMPLETENESS, 0.2)).weight
        
        return QualityMetric(
            metric_type=MetricType.COMPLETENESS,
            value=base_completeness,
            confidence=confidence,
            weight=weight,
            normalized_value=base_completeness,  # Will be updated later
            sub_metrics=sub_metrics,
            contributing_factors=contributing_factors,
            calculation_method="profile_based"
        )
    
    def _calculate_accuracy_metric(self, data_profile: DataQualityProfile) -> QualityMetric:
        """Calculate accuracy metric."""
        # Get accuracy from profile
        base_accuracy = data_profile.accuracy_score
        
        # Calculate sub-metrics
        sub_metrics = {}
        contributing_factors = {}
        
        # Data type accuracy
        if hasattr(data_profile, 'column_profiles'):
            type_accuracy_scores = []
            for column_name, column_profile in data_profile.column_profiles.items():
                if hasattr(column_profile, 'data_type_accuracy'):
                    type_accuracy = column_profile.data_type_accuracy
                    type_accuracy_scores.append(type_accuracy)
                    sub_metrics[f'column_{column_name}_type_accuracy'] = type_accuracy
            
            if type_accuracy_scores:
                sub_metrics['average_type_accuracy'] = np.mean(type_accuracy_scores)
                contributing_factors['type_accuracy_variance'] = np.var(type_accuracy_scores)
        
        # Domain-specific accuracy
        if hasattr(data_profile, 'domain_accuracy'):
            sub_metrics['domain_accuracy'] = data_profile.domain_accuracy
            contributing_factors['domain_validation_rules_count'] = getattr(data_profile, 'domain_rules_count', 0)
        
        # Pattern accuracy
        if hasattr(data_profile, 'pattern_accuracy'):
            sub_metrics['pattern_accuracy'] = data_profile.pattern_accuracy
        
        # Calculate confidence
        confidence = 0.8  # Default confidence for accuracy
        
        # Get weight
        weight = self._metric_weights.get(MetricType.ACCURACY, MetricWeight(MetricType.ACCURACY, 0.25)).weight
        
        return QualityMetric(
            metric_type=MetricType.ACCURACY,
            value=base_accuracy,
            confidence=confidence,
            weight=weight,
            normalized_value=base_accuracy,
            sub_metrics=sub_metrics,
            contributing_factors=contributing_factors,
            calculation_method="profile_based"
        )
    
    def _calculate_consistency_metric(self, data_profile: DataQualityProfile) -> QualityMetric:
        """Calculate consistency metric."""
        # Get consistency from profile
        base_consistency = data_profile.consistency_score
        
        # Calculate sub-metrics
        sub_metrics = {}
        contributing_factors = {}
        
        # Format consistency
        if hasattr(data_profile, 'format_consistency'):
            sub_metrics['format_consistency'] = data_profile.format_consistency
        
        # Value consistency
        if hasattr(data_profile, 'value_consistency'):
            sub_metrics['value_consistency'] = data_profile.value_consistency
        
        # Cross-column consistency
        if hasattr(data_profile, 'cross_column_consistency'):
            sub_metrics['cross_column_consistency'] = data_profile.cross_column_consistency
        
        # Calculate statistical consistency
        if hasattr(data_profile, 'column_profiles'):
            value_distributions = []
            for column_profile in data_profile.column_profiles.values():
                if hasattr(column_profile, 'value_distribution'):
                    value_distributions.append(column_profile.value_distribution)
            
            if value_distributions:
                # Calculate distribution consistency
                contributing_factors['distribution_consistency'] = self._calculate_distribution_consistency(value_distributions)
        
        # Calculate confidence
        confidence = 0.75
        
        # Get weight
        weight = self._metric_weights.get(MetricType.CONSISTENCY, MetricWeight(MetricType.CONSISTENCY, 0.15)).weight
        
        return QualityMetric(
            metric_type=MetricType.CONSISTENCY,
            value=base_consistency,
            confidence=confidence,
            weight=weight,
            normalized_value=base_consistency,
            sub_metrics=sub_metrics,
            contributing_factors=contributing_factors,
            calculation_method="profile_based"
        )
    
    def _calculate_validity_metric(self, data_profile: DataQualityProfile) -> QualityMetric:
        """Calculate validity metric."""
        # Get validity from profile
        base_validity = data_profile.validity_score
        
        # Calculate sub-metrics
        sub_metrics = {}
        contributing_factors = {}
        
        # Constraint validity
        if hasattr(data_profile, 'constraint_violations'):
            total_constraints = getattr(data_profile, 'total_constraints', 1)
            violations = len(data_profile.constraint_violations)
            constraint_validity = max(0, 1 - (violations / total_constraints))
            sub_metrics['constraint_validity'] = constraint_validity
            contributing_factors['constraint_violations_count'] = violations
        
        # Business rule validity
        if hasattr(data_profile, 'business_rule_violations'):
            total_rules = getattr(data_profile, 'total_business_rules', 1)
            violations = len(data_profile.business_rule_violations)
            business_rule_validity = max(0, 1 - (violations / total_rules))
            sub_metrics['business_rule_validity'] = business_rule_validity
            contributing_factors['business_rule_violations_count'] = violations
        
        # Schema validity
        if hasattr(data_profile, 'schema_compliance'):
            sub_metrics['schema_validity'] = data_profile.schema_compliance
        
        # Calculate confidence
        confidence = 0.85
        
        # Get weight
        weight = self._metric_weights.get(MetricType.VALIDITY, MetricWeight(MetricType.VALIDITY, 0.2)).weight
        
        return QualityMetric(
            metric_type=MetricType.VALIDITY,
            value=base_validity,
            confidence=confidence,
            weight=weight,
            normalized_value=base_validity,
            sub_metrics=sub_metrics,
            contributing_factors=contributing_factors,
            calculation_method="profile_based"
        )
    
    def _calculate_uniqueness_metric(self, data_profile: DataQualityProfile) -> QualityMetric:
        """Calculate uniqueness metric."""
        # Get uniqueness from profile
        base_uniqueness = data_profile.uniqueness_score
        
        # Calculate sub-metrics
        sub_metrics = {}
        contributing_factors = {}
        
        # Duplicate analysis
        if hasattr(data_profile, 'duplicate_count'):
            total_rows = data_profile.row_count
            duplicates = data_profile.duplicate_count
            
            if total_rows > 0:
                duplicate_rate = duplicates / total_rows
                sub_metrics['duplicate_rate'] = duplicate_rate
                contributing_factors['duplicate_count'] = duplicates
                contributing_factors['total_rows'] = total_rows
        
        # Key uniqueness
        if hasattr(data_profile, 'key_uniqueness'):
            sub_metrics['key_uniqueness'] = data_profile.key_uniqueness
        
        # Column-level uniqueness
        if hasattr(data_profile, 'column_profiles'):
            column_uniqueness = []
            for column_name, column_profile in data_profile.column_profiles.items():
                if hasattr(column_profile, 'uniqueness'):
                    uniqueness = column_profile.uniqueness
                    column_uniqueness.append(uniqueness)
                    sub_metrics[f'column_{column_name}_uniqueness'] = uniqueness
            
            if column_uniqueness:
                contributing_factors['column_uniqueness_variance'] = np.var(column_uniqueness)
        
        # Calculate confidence
        confidence = 0.9
        
        # Get weight
        weight = self._metric_weights.get(MetricType.UNIQUENESS, MetricWeight(MetricType.UNIQUENESS, 0.1)).weight
        
        return QualityMetric(
            metric_type=MetricType.UNIQUENESS,
            value=base_uniqueness,
            confidence=confidence,
            weight=weight,
            normalized_value=base_uniqueness,
            sub_metrics=sub_metrics,
            contributing_factors=contributing_factors,
            calculation_method="profile_based"
        )
    
    def _calculate_timeliness_metric(self, data_profile: DataQualityProfile) -> QualityMetric:
        """Calculate timeliness metric."""
        # Get timeliness from profile or calculate default
        base_timeliness = getattr(data_profile, 'timeliness_score', 0.8)
        
        # Calculate sub-metrics
        sub_metrics = {}
        contributing_factors = {}
        
        # Freshness analysis
        if hasattr(data_profile, 'last_updated'):
            last_updated = data_profile.last_updated
            age_hours = (datetime.now() - last_updated).total_seconds() / 3600
            
            # Calculate freshness score (exponential decay)
            freshness = np.exp(-age_hours / 24)  # Decay over 24 hours
            sub_metrics['freshness'] = freshness
            contributing_factors['age_hours'] = age_hours
        
        # Update frequency
        if hasattr(data_profile, 'update_frequency_hours'):
            expected_frequency = data_profile.update_frequency_hours
            actual_frequency = contributing_factors.get('age_hours', 0)
            
            if expected_frequency > 0:
                frequency_score = min(1.0, expected_frequency / max(actual_frequency, 1))
                sub_metrics['frequency_score'] = frequency_score
                contributing_factors['expected_frequency'] = expected_frequency
        
        # Data latency
        if hasattr(data_profile, 'data_latency_hours'):
            latency = data_profile.data_latency_hours
            # Lower latency is better
            latency_score = max(0, 1 - (latency / 24))  # Normalize to 24 hours
            sub_metrics['latency_score'] = latency_score
            contributing_factors['data_latency_hours'] = latency
        
        # Calculate confidence
        confidence = 0.7
        
        # Get weight
        weight = self._metric_weights.get(MetricType.TIMELINESS, MetricWeight(MetricType.TIMELINESS, 0.1)).weight
        
        return QualityMetric(
            metric_type=MetricType.TIMELINESS,
            value=base_timeliness,
            confidence=confidence,
            weight=weight,
            normalized_value=base_timeliness,
            sub_metrics=sub_metrics,
            contributing_factors=contributing_factors,
            calculation_method="profile_based"
        )
    
    def _calculate_overall_score(self,
                               metrics: Dict[MetricType, QualityMetric],
                               scoring_method: ScoringMethod) -> float:
        """Calculate overall quality score using specified method."""
        if not metrics:
            return 0.0
        
        values = [metric.value for metric in metrics.values()]
        weights = [metric.weight for metric in metrics.values()]
        
        if scoring_method == ScoringMethod.WEIGHTED_AVERAGE:
            return np.average(values, weights=weights)
        
        elif scoring_method == ScoringMethod.GEOMETRIC_MEAN:
            # Weighted geometric mean
            weighted_product = 1.0
            total_weight = sum(weights)
            for value, weight in zip(values, weights):
                weighted_product *= (value ** (weight / total_weight))
            return weighted_product
        
        elif scoring_method == ScoringMethod.HARMONIC_MEAN:
            # Weighted harmonic mean
            numerator = sum(weights)
            denominator = sum(weight / max(value, 0.001) for value, weight in zip(values, weights))
            return numerator / denominator if denominator > 0 else 0.0
        
        elif scoring_method == ScoringMethod.MIN_MAX_NORMALIZATION:
            # Normalize then take weighted average
            min_val = min(values)
            max_val = max(values)
            if max_val > min_val:
                normalized = [(v - min_val) / (max_val - min_val) for v in values]
            else:
                normalized = values
            return np.average(normalized, weights=weights)
        
        elif scoring_method == ScoringMethod.MULTI_CRITERIA:
            # Multi-criteria decision analysis
            return self._calculate_multi_criteria_score(metrics)
        
        else:
            # Default to weighted average
            return np.average(values, weights=weights)
    
    def _calculate_multi_criteria_score(self, metrics: Dict[MetricType, QualityMetric]) -> float:
        """Calculate multi-criteria decision analysis score."""
        if not metrics:
            return 0.0
        
        # Apply different aggregation rules for different quality aspects
        critical_metrics = [MetricType.ACCURACY, MetricType.VALIDITY, MetricType.COMPLETENESS]
        enhancement_metrics = [MetricType.CONSISTENCY, MetricType.UNIQUENESS, MetricType.TIMELINESS]
        
        # Critical metrics use minimum (weakest link)
        critical_values = [metrics[mt].value for mt in critical_metrics if mt in metrics]
        critical_weights = [metrics[mt].weight for mt in critical_metrics if mt in metrics]
        
        # Enhancement metrics use weighted average
        enhancement_values = [metrics[mt].value for mt in enhancement_metrics if mt in metrics]
        enhancement_weights = [metrics[mt].weight for mt in enhancement_metrics if mt in metrics]
        
        # Calculate component scores
        critical_score = min(critical_values) if critical_values else 1.0
        enhancement_score = np.average(enhancement_values, weights=enhancement_weights) if enhancement_values else 1.0
        
        # Combine with different weights
        combined_score = (critical_score * 0.7) + (enhancement_score * 0.3)
        
        return combined_score
    
    def _normalize_score(self, score: float, metrics: Dict[MetricType, QualityMetric]) -> float:
        """Normalize score using z-score if enabled."""
        if not self.config.enable_z_score_normalization:
            return score
        
        # Use historical data if available
        historical_scores = []
        for metric_list in self._historical_scores.values():
            historical_scores.extend([s.overall_score for s in metric_list])
        
        if len(historical_scores) < 5:
            return score
        
        # Calculate z-score
        mean_score = np.mean(historical_scores)
        std_score = np.std(historical_scores)
        
        if std_score > 0:
            z_score = (score - mean_score) / std_score
            # Convert z-score to 0-1 range
            normalized = 0.5 + (z_score / 6)  # Assuming 6 sigma range
            return max(0.0, min(1.0, normalized))
        
        return score
    
    def _normalize_metric_value(self, value: float) -> float:
        """Normalize individual metric value."""
        return max(0.0, min(1.0, value))
    
    def _determine_quality_tier(self, score: float) -> QualityTier:
        """Determine quality tier based on score."""
        if score >= 0.95:
            return QualityTier.EXCELLENT
        elif score >= 0.8:
            return QualityTier.GOOD
        elif score >= 0.6:
            return QualityTier.FAIR
        elif score >= 0.4:
            return QualityTier.POOR
        else:
            return QualityTier.CRITICAL
    
    def _calculate_confidence_interval(self, metrics: Dict[MetricType, QualityMetric]) -> Tuple[float, float]:
        """Calculate confidence interval for overall score."""
        if not metrics:
            return (0.0, 0.0)
        
        # Use metric confidences to estimate overall confidence
        confidences = [metric.confidence for metric in metrics.values()]
        values = [metric.value for metric in metrics.values()]
        weights = [metric.weight for metric in metrics.values()]
        
        # Calculate weighted average and variance
        weighted_avg = np.average(values, weights=weights)
        weighted_var = np.average((np.array(values) - weighted_avg) ** 2, weights=weights)
        
        # Adjust by confidence
        avg_confidence = np.mean(confidences)
        adjusted_std = np.sqrt(weighted_var) * (1 - avg_confidence)
        
        # Calculate confidence interval
        alpha = self.config.confidence_interval_alpha
        z_score = 1.96  # 95% confidence interval
        
        margin = z_score * adjusted_std
        lower_bound = max(0.0, weighted_avg - margin)
        upper_bound = min(1.0, weighted_avg + margin)
        
        return (lower_bound, upper_bound)
    
    def _perform_benchmarking(self, dataset_id: str, score: float) -> Tuple[Dict[str, float], Optional[float]]:
        """Perform benchmarking analysis."""
        benchmark_scores = {}
        percentile_ranking = None
        
        # Get historical scores for this dataset
        historical_scores = [s.overall_score for s in self._historical_scores.get(dataset_id, [])]
        
        if len(historical_scores) >= 5:
            # Calculate percentiles
            percentiles = self.config.benchmark_percentiles
            
            for p in percentiles:
                benchmark_scores[f'p{p}'] = np.percentile(historical_scores, p)
            
            # Calculate current score's percentile ranking
            percentile_ranking = (sum(1 for s in historical_scores if s <= score) / len(historical_scores)) * 100
        
        # Get cross-dataset benchmarks
        all_scores = []
        for scores_list in self._historical_scores.values():
            all_scores.extend([s.overall_score for s in scores_list])
        
        if len(all_scores) >= 10:
            for p in self.config.benchmark_percentiles:
                benchmark_scores[f'industry_p{p}'] = np.percentile(all_scores, p)
        
        return benchmark_scores, percentile_ranking
    
    def _analyze_temporal_trends(self, historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze temporal trends in quality data."""
        if not historical_data or len(historical_data) < 3:
            return {}
        
        # Extract timestamps and scores
        timestamps = [datetime.fromisoformat(d['timestamp']) if isinstance(d['timestamp'], str) 
                     else d['timestamp'] for d in historical_data]
        scores = [d['quality_score'] for d in historical_data]
        
        # Sort by timestamp
        sorted_data = sorted(zip(timestamps, scores))
        timestamps, scores = zip(*sorted_data)
        
        # Calculate trend
        x = np.arange(len(scores))
        slope, intercept = np.polyfit(x, scores, 1)
        
        # Calculate trend significance
        correlation = np.corrcoef(x, scores)[0, 1]
        
        # Determine trend direction
        if abs(slope) < 0.001:
            trend_direction = "stable"
        elif slope > 0:
            trend_direction = "improving"
        else:
            trend_direction = "declining"
        
        # Calculate volatility
        volatility = np.std(scores)
        
        # Recent vs historical comparison
        recent_avg = np.mean(scores[-7:]) if len(scores) >= 7 else np.mean(scores)
        historical_avg = np.mean(scores[:-7]) if len(scores) >= 14 else np.mean(scores)
        
        return {
            'trend_direction': trend_direction,
            'trend_slope': slope,
            'trend_significance': abs(correlation),
            'volatility': volatility,
            'recent_average': recent_avg,
            'historical_average': historical_avg,
            'improvement_rate': (recent_avg - historical_avg) / historical_avg if historical_avg > 0 else 0
        }
    
    def _compare_with_historical(self, current_score: float, dataset_id: str) -> Dict[str, Any]:
        """Compare current score with historical data."""
        historical_scores = [s.overall_score for s in self._historical_scores.get(dataset_id, [])]
        
        if not historical_scores:
            return {}
        
        avg_score = np.mean(historical_scores)
        best_score = max(historical_scores)
        worst_score = min(historical_scores)
        
        return {
            'current_vs_average': current_score - avg_score,
            'current_vs_best': current_score - best_score,
            'current_vs_worst': current_score - worst_score,
            'percentile_rank': (sum(1 for s in historical_scores if s <= current_score) / len(historical_scores)) * 100,
            'is_best_ever': current_score >= best_score,
            'is_worst_ever': current_score <= worst_score
        }
    
    def _calculate_business_impact(self, metrics: Dict[MetricType, QualityMetric]) -> float:
        """Calculate business impact score."""
        if not self.config.enable_business_impact_weighting:
            return None
        
        # Business impact weights for different metrics
        business_weights = {
            MetricType.ACCURACY: 0.3,
            MetricType.COMPLETENESS: 0.25,
            MetricType.VALIDITY: 0.2,
            MetricType.TIMELINESS: 0.15,
            MetricType.CONSISTENCY: 0.1
        }
        
        impact_score = 0.0
        total_weight = 0.0
        
        for metric_type, metric in metrics.items():
            if metric_type in business_weights:
                weight = business_weights[metric_type]
                # Lower quality has higher business impact (negative)
                impact_score += (1 - metric.value) * weight
                total_weight += weight
        
        return impact_score / total_weight if total_weight > 0 else 0.0
    
    def _identify_critical_issues(self, metrics: Dict[MetricType, QualityMetric]) -> List[str]:
        """Identify critical quality issues."""
        issues = []
        
        for metric_type, metric in metrics.items():
            if metric.is_critical():
                issues.append(f"Critical {metric_type.value} issue: {metric.value:.2%}")
            
            # Check specific thresholds
            weight_info = self._metric_weights.get(metric_type)
            if weight_info and metric.value < weight_info.critical_threshold:
                issues.append(f"{metric_type.value} below critical threshold: {metric.value:.2%} < {weight_info.critical_threshold:.2%}")
        
        return issues
    
    def _generate_improvement_recommendations(self, metrics: Dict[MetricType, QualityMetric]) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        # Get lowest performing metrics
        sorted_metrics = sorted(metrics.items(), key=lambda x: x[1].value)
        
        for metric_type, metric in sorted_metrics[:3]:  # Top 3 improvement areas
            if metric.value < 0.8:  # Below good threshold
                if metric_type == MetricType.COMPLETENESS:
                    recommendations.append("Improve data completeness by addressing missing values and implementing data validation")
                elif metric_type == MetricType.ACCURACY:
                    recommendations.append("Enhance data accuracy through improved validation rules and data cleansing")
                elif metric_type == MetricType.CONSISTENCY:
                    recommendations.append("Standardize data formats and implement consistency checks across systems")
                elif metric_type == MetricType.VALIDITY:
                    recommendations.append("Strengthen business rule validation and constraint enforcement")
                elif metric_type == MetricType.UNIQUENESS:
                    recommendations.append("Implement duplicate detection and prevention mechanisms")
                elif metric_type == MetricType.TIMELINESS:
                    recommendations.append("Optimize data refresh cycles and reduce processing latency")
        
        return recommendations
    
    def _calculate_distribution_consistency(self, distributions: List[Dict[str, Any]]) -> float:
        """Calculate consistency across value distributions."""
        if not distributions:
            return 1.0
        
        # Simplified consistency calculation
        # In practice, this would involve statistical tests
        return 0.8  # Placeholder
    
    def get_quality_dashboard_data(self, dataset_ids: List[str] = None) -> Dict[str, Any]:
        """Get quality dashboard data for multiple datasets.
        
        Args:
            dataset_ids: Specific datasets to include
            
        Returns:
            Dashboard data
        """
        if dataset_ids is None:
            dataset_ids = list(self._historical_scores.keys())
        
        dashboard_data = {
            'summary': {
                'total_datasets': len(dataset_ids),
                'datasets_monitored': len([d for d in dataset_ids if d in self._historical_scores]),
                'avg_quality_score': 0.0,
                'critical_datasets': 0,
                'improvement_trend': 0.0
            },
            'datasets': {},
            'benchmarks': {},
            'trends': {}
        }
        
        all_scores = []
        critical_count = 0
        
        for dataset_id in dataset_ids:
            if dataset_id in self._historical_scores:
                scores = self._historical_scores[dataset_id]
                if scores:
                    latest_score = scores[-1]
                    all_scores.append(latest_score.overall_score)
                    
                    if latest_score.quality_tier == QualityTier.CRITICAL:
                        critical_count += 1
                    
                    dashboard_data['datasets'][dataset_id] = {
                        'current_score': latest_score.overall_score,
                        'quality_tier': latest_score.quality_tier.value,
                        'critical_issues': len(latest_score.critical_issues),
                        'trend_direction': latest_score.trend_analysis.get('trend_direction', 'stable'),
                        'last_updated': latest_score.calculation_timestamp.isoformat()
                    }
        
        # Calculate summary statistics
        if all_scores:
            dashboard_data['summary']['avg_quality_score'] = np.mean(all_scores)
            dashboard_data['summary']['critical_datasets'] = critical_count
            
            # Calculate improvement trend
            if len(all_scores) >= 2:
                recent_avg = np.mean(all_scores[-5:])
                historical_avg = np.mean(all_scores[:-5]) if len(all_scores) > 5 else np.mean(all_scores)
                dashboard_data['summary']['improvement_trend'] = (recent_avg - historical_avg) / historical_avg if historical_avg > 0 else 0
        
        # Add benchmark data
        dashboard_data['benchmarks'] = {
            'industry_average': np.mean(all_scores) if all_scores else 0.0,
            'top_quartile': np.percentile(all_scores, 75) if all_scores else 0.0,
            'bottom_quartile': np.percentile(all_scores, 25) if all_scores else 0.0
        }
        
        return dashboard_data