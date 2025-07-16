"""Comprehensive Quality Scoring Engine.

Enhanced quality scoring system that provides detailed, multi-dimensional quality assessment
with configurable scoring algorithms, quality dimensions, and enterprise-grade reporting.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging
from enum import Enum
from abc import ABC, abstractmethod
import math
from statistics import mean, median, stdev

from ...domain.entities.quality_profile import DataQualityProfile, DatasetId
from ...domain.entities.quality_scores import QualityScores
from ...domain.entities.validation_rule import ValidationResult, ValidationStatus
from ...domain.entities.quality_issue import QualityIssue, IssueSeverity
from .advanced_quality_metrics_service import AdvancedQualityScore, MetricType

logger = logging.getLogger(__name__)


class QualityDimension(Enum):
    """Quality dimensions for comprehensive assessment."""
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    VALIDITY = "validity"
    UNIQUENESS = "uniqueness"
    TIMELINESS = "timeliness"
    CONFORMITY = "conformity"
    INTEGRITY = "integrity"
    RELIABILITY = "reliability"
    RELEVANCE = "relevance"


class ScoringAlgorithm(Enum):
    """Scoring algorithms for quality assessment."""
    WEIGHTED_AVERAGE = "weighted_average"
    GEOMETRIC_MEAN = "geometric_mean"
    HARMONIC_MEAN = "harmonic_mean"
    MULTIPLICATIVE = "multiplicative"
    MINIMUM = "minimum"
    FUZZY_LOGIC = "fuzzy_logic"
    BAYESIAN = "bayesian"
    NEURAL_NETWORK = "neural_network"


class QualityLevel(Enum):
    """Quality levels for classification."""
    EXCELLENT = "excellent"      # 95-100%
    VERY_GOOD = "very_good"     # 90-94%
    GOOD = "good"               # 80-89%
    SATISFACTORY = "satisfactory" # 70-79%
    NEEDS_IMPROVEMENT = "needs_improvement" # 60-69%
    POOR = "poor"               # 40-59%
    VERY_POOR = "very_poor"     # 20-39%
    CRITICAL = "critical"       # 0-19%


@dataclass(frozen=True)
class QualityWeight:
    """Weight configuration for quality dimensions."""
    dimension: QualityDimension
    weight: float
    min_threshold: float = 0.0
    max_threshold: float = 1.0
    critical_threshold: float = 0.5
    
    def __post_init__(self):
        if not 0 <= self.weight <= 1:
            raise ValueError("Weight must be between 0 and 1")
        if not 0 <= self.min_threshold <= self.max_threshold <= 1:
            raise ValueError("Invalid threshold values")


@dataclass(frozen=True)
class QualityContext:
    """Context information for quality assessment."""
    business_domain: str = "general"
    data_sensitivity: str = "medium"  # low, medium, high, critical
    regulatory_requirements: List[str] = field(default_factory=list)
    usage_patterns: Dict[str, Any] = field(default_factory=dict)
    stakeholder_expectations: Dict[str, float] = field(default_factory=dict)
    
    def get_sensitivity_multiplier(self) -> float:
        """Get scoring multiplier based on data sensitivity."""
        multipliers = {
            'low': 1.0,
            'medium': 1.1,
            'high': 1.2,
            'critical': 1.3
        }
        return multipliers.get(self.data_sensitivity, 1.0)


@dataclass
class DimensionScore:
    """Score for a specific quality dimension."""
    dimension: QualityDimension
    score: float
    weighted_score: float
    weight: float
    confidence: float
    
    # Detailed breakdown
    sub_scores: Dict[str, float] = field(default_factory=dict)
    contributing_factors: Dict[str, float] = field(default_factory=dict)
    quality_issues: List[QualityIssue] = field(default_factory=list)
    
    # Recommendations
    improvement_recommendations: List[str] = field(default_factory=list)
    
    # Metadata
    calculation_method: str = "default"
    calculation_timestamp: datetime = field(default_factory=datetime.now)
    
    def get_quality_level(self) -> QualityLevel:
        """Get quality level based on score."""
        if self.score >= 0.95:
            return QualityLevel.EXCELLENT
        elif self.score >= 0.90:
            return QualityLevel.VERY_GOOD
        elif self.score >= 0.80:
            return QualityLevel.GOOD
        elif self.score >= 0.70:
            return QualityLevel.SATISFACTORY
        elif self.score >= 0.60:
            return QualityLevel.NEEDS_IMPROVEMENT
        elif self.score >= 0.40:
            return QualityLevel.POOR
        elif self.score >= 0.20:
            return QualityLevel.VERY_POOR
        else:
            return QualityLevel.CRITICAL
    
    def is_critical(self) -> bool:
        """Check if dimension is in critical state."""
        return self.score < 0.2 or self.get_quality_level() == QualityLevel.CRITICAL
    
    def get_score_summary(self) -> Dict[str, Any]:
        """Get dimension score summary."""
        return {
            'dimension': self.dimension.value,
            'score': round(self.score, 4),
            'weighted_score': round(self.weighted_score, 4),
            'weight': self.weight,
            'confidence': round(self.confidence, 3),
            'quality_level': self.get_quality_level().value,
            'is_critical': self.is_critical(),
            'issues_count': len(self.quality_issues),
            'recommendations_count': len(self.improvement_recommendations),
            'calculation_method': self.calculation_method,
            'calculation_timestamp': self.calculation_timestamp.isoformat()
        }


@dataclass
class ComprehensiveQualityScore:
    """Comprehensive quality score with detailed analysis."""
    dataset_id: str
    overall_score: float
    weighted_score: float
    quality_level: QualityLevel
    confidence: float
    
    # Dimensional scores
    dimension_scores: Dict[QualityDimension, DimensionScore]
    
    # Scoring metadata
    scoring_algorithm: ScoringAlgorithm
    quality_context: QualityContext
    weights_used: Dict[QualityDimension, float]
    
    # Analysis results
    critical_dimensions: List[QualityDimension] = field(default_factory=list)
    improvement_opportunities: List[Tuple[QualityDimension, float]] = field(default_factory=list)
    
    # Benchmarking
    benchmark_scores: Dict[str, float] = field(default_factory=dict)
    percentile_ranking: Optional[float] = None
    
    # Trends and predictions
    trend_analysis: Dict[str, Any] = field(default_factory=dict)
    predicted_scores: Dict[str, float] = field(default_factory=dict)
    
    # Business impact
    business_impact_score: float = 0.0
    risk_assessment: Dict[str, Any] = field(default_factory=dict)
    
    # Compliance
    regulatory_compliance: Dict[str, bool] = field(default_factory=dict)
    
    # Metadata
    calculation_timestamp: datetime = field(default_factory=datetime.now)
    calculation_duration_ms: float = 0.0
    
    def get_critical_dimensions(self) -> List[DimensionScore]:
        """Get dimensions in critical state."""
        return [score for score in self.dimension_scores.values() if score.is_critical()]
    
    def get_top_improvement_opportunities(self, n: int = 5) -> List[Tuple[QualityDimension, float]]:
        """Get top improvement opportunities."""
        opportunities = []
        for dimension, score in self.dimension_scores.items():
            # Calculate improvement potential (weight * (1 - score))
            potential = score.weight * (1 - score.score)
            opportunities.append((dimension, potential))
        
        # Sort by potential impact
        opportunities.sort(key=lambda x: x[1], reverse=True)
        return opportunities[:n]
    
    def get_quality_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive quality dashboard data."""
        return {
            'dataset_id': self.dataset_id,
            'overall_assessment': {
                'score': round(self.overall_score, 4),
                'weighted_score': round(self.weighted_score, 4),
                'quality_level': self.quality_level.value,
                'confidence': round(self.confidence, 3),
                'scoring_algorithm': self.scoring_algorithm.value
            },
            'dimensional_breakdown': {
                dim.value: score.get_score_summary()
                for dim, score in self.dimension_scores.items()
            },
            'critical_analysis': {
                'critical_dimensions': [dim.value for dim in self.get_critical_dimensions()],
                'improvement_opportunities': [
                    {'dimension': dim.value, 'potential': round(potential, 4)}
                    for dim, potential in self.get_top_improvement_opportunities()
                ]
            },
            'business_context': {
                'business_impact_score': round(self.business_impact_score, 4),
                'risk_assessment': self.risk_assessment,
                'regulatory_compliance': self.regulatory_compliance
            },
            'benchmarking': {
                'benchmark_scores': self.benchmark_scores,
                'percentile_ranking': self.percentile_ranking
            },
            'trends': self.trend_analysis,
            'metadata': {
                'calculation_timestamp': self.calculation_timestamp.isoformat(),
                'calculation_duration_ms': self.calculation_duration_ms,
                'quality_context': {
                    'business_domain': self.quality_context.business_domain,
                    'data_sensitivity': self.quality_context.data_sensitivity,
                    'regulatory_requirements': self.quality_context.regulatory_requirements
                }
            }
        }


class QualityDimensionCalculator(ABC):
    """Abstract base class for quality dimension calculators."""
    
    @abstractmethod
    def calculate(self, df: pd.DataFrame, 
                 validation_results: List[ValidationResult],
                 context: QualityContext) -> DimensionScore:
        """Calculate dimension score."""
        pass
    
    @abstractmethod
    def get_dimension(self) -> QualityDimension:
        """Get the quality dimension this calculator handles."""
        pass


class CompletenessCalculator(QualityDimensionCalculator):
    """Calculator for completeness dimension."""
    
    def get_dimension(self) -> QualityDimension:
        return QualityDimension.COMPLETENESS
    
    def calculate(self, df: pd.DataFrame, 
                 validation_results: List[ValidationResult],
                 context: QualityContext) -> DimensionScore:
        """Calculate completeness score."""
        if len(df) == 0:
            return DimensionScore(
                dimension=QualityDimension.COMPLETENESS,
                score=0.0,
                weighted_score=0.0,
                weight=1.0,
                confidence=0.0,
                calculation_method="empty_dataset"
            )
        
        total_cells = len(df) * len(df.columns)
        
        # Calculate null percentages
        null_counts = df.isnull().sum()
        total_nulls = null_counts.sum()
        
        # Overall completeness
        overall_completeness = 1 - (total_nulls / total_cells) if total_cells > 0 else 0
        
        # Column-level completeness
        column_completeness = {}
        for col in df.columns:
            col_completeness = 1 - (null_counts[col] / len(df))
            column_completeness[col] = col_completeness
        
        # Calculate weighted completeness (critical columns weighted more)
        critical_columns = context.usage_patterns.get('critical_columns', [])
        weighted_completeness = 0.0
        total_weight = 0.0
        
        for col in df.columns:
            weight = 2.0 if col in critical_columns else 1.0
            weighted_completeness += column_completeness[col] * weight
            total_weight += weight
        
        if total_weight > 0:
            weighted_completeness /= total_weight
        else:
            weighted_completeness = overall_completeness
        
        # Calculate confidence based on data volume
        confidence = min(1.0, np.log10(max(len(df), 1)) / 5)
        
        # Generate recommendations
        recommendations = []
        low_completeness_cols = [col for col, comp in column_completeness.items() if comp < 0.8]
        if low_completeness_cols:
            recommendations.append(f"Improve completeness for columns: {', '.join(low_completeness_cols[:5])}")
        
        if overall_completeness < 0.9:
            recommendations.append("Consider data collection improvements to reduce missing values")
        
        return DimensionScore(
            dimension=QualityDimension.COMPLETENESS,
            score=overall_completeness,
            weighted_score=weighted_completeness,
            weight=1.0,
            confidence=confidence,
            sub_scores=column_completeness,
            contributing_factors={
                'total_cells': total_cells,
                'null_cells': total_nulls,
                'critical_columns_count': len(critical_columns)
            },
            improvement_recommendations=recommendations,
            calculation_method="null_analysis"
        )


class AccuracyCalculator(QualityDimensionCalculator):
    """Calculator for accuracy dimension."""
    
    def get_dimension(self) -> QualityDimension:
        return QualityDimension.ACCURACY
    
    def calculate(self, df: pd.DataFrame, 
                 validation_results: List[ValidationResult],
                 context: QualityContext) -> DimensionScore:
        """Calculate accuracy score."""
        if len(df) == 0:
            return DimensionScore(
                dimension=QualityDimension.ACCURACY,
                score=0.0,
                weighted_score=0.0,
                weight=1.0,
                confidence=0.0,
                calculation_method="empty_dataset"
            )
        
        # Accuracy from validation results
        accuracy_results = [r for r in validation_results 
                          if hasattr(r, 'rule_id') and 'accuracy' in str(r.rule_id).lower()]
        
        if accuracy_results:
            accuracy_scores = []
            for result in accuracy_results:
                if result.total_records > 0:
                    accuracy_score = result.passed_records / result.total_records
                    accuracy_scores.append(accuracy_score)
            
            overall_accuracy = np.mean(accuracy_scores) if accuracy_scores else 0.8
        else:
            # Estimate accuracy based on data characteristics
            overall_accuracy = self._estimate_accuracy(df)
        
        # Calculate data type accuracy
        type_accuracy = self._calculate_type_accuracy(df)
        
        # Calculate format accuracy
        format_accuracy = self._calculate_format_accuracy(df)
        
        # Calculate domain accuracy
        domain_accuracy = self._calculate_domain_accuracy(df, context)
        
        # Combined accuracy score
        combined_accuracy = np.mean([overall_accuracy, type_accuracy, format_accuracy, domain_accuracy])
        
        # Calculate confidence
        confidence = 0.8 if accuracy_results else 0.6
        
        # Generate recommendations
        recommendations = []
        if type_accuracy < 0.9:
            recommendations.append("Review data types and fix conversion errors")
        if format_accuracy < 0.9:
            recommendations.append("Standardize data formats and patterns")
        if domain_accuracy < 0.9:
            recommendations.append("Validate domain-specific rules and constraints")
        
        return DimensionScore(
            dimension=QualityDimension.ACCURACY,
            score=combined_accuracy,
            weighted_score=combined_accuracy,
            weight=1.0,
            confidence=confidence,
            sub_scores={
                'overall_accuracy': overall_accuracy,
                'type_accuracy': type_accuracy,
                'format_accuracy': format_accuracy,
                'domain_accuracy': domain_accuracy
            },
            improvement_recommendations=recommendations,
            calculation_method="validation_based"
        )
    
    def _estimate_accuracy(self, df: pd.DataFrame) -> float:
        """Estimate accuracy based on data characteristics."""
        # Simple heuristic based on data quality indicators
        accuracy_indicators = []
        
        # Check for obvious errors in numeric columns
        for col in df.select_dtypes(include=[np.number]).columns:
            if len(df[col].dropna()) > 0:
                # Check for extreme values
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                if IQR > 0:
                    outliers = ((df[col] < Q1 - 3 * IQR) | (df[col] > Q3 + 3 * IQR)).sum()
                    accuracy_indicators.append(1 - (outliers / len(df)))
                else:
                    accuracy_indicators.append(0.9)
        
        # Check for obvious errors in string columns
        for col in df.select_dtypes(include=['object']).columns:
            if len(df[col].dropna()) > 0:
                # Check for suspiciously short/long strings
                str_lengths = df[col].astype(str).str.len()
                if len(str_lengths) > 0:
                    mean_length = str_lengths.mean()
                    if mean_length > 0:
                        extreme_lengths = ((str_lengths < mean_length * 0.1) | 
                                         (str_lengths > mean_length * 5)).sum()
                        accuracy_indicators.append(1 - (extreme_lengths / len(df)))
                    else:
                        accuracy_indicators.append(0.8)
        
        return np.mean(accuracy_indicators) if accuracy_indicators else 0.8
    
    def _calculate_type_accuracy(self, df: pd.DataFrame) -> float:
        """Calculate data type accuracy."""
        type_accuracy_scores = []
        
        for col in df.columns:
            # Check if column values match expected type
            if df[col].dtype == 'object':
                # For object columns, check if they can be converted to numeric if they look numeric
                non_null_values = df[col].dropna()
                if len(non_null_values) > 0:
                    # Try to convert to numeric
                    try:
                        pd.to_numeric(non_null_values)
                        # If successful, check if it should be numeric
                        numeric_ratio = non_null_values.str.match(r'^-?\d+\.?\d*$').mean()
                        if numeric_ratio > 0.8:  # Looks like it should be numeric
                            type_accuracy_scores.append(0.7)  # Penalize for wrong type
                        else:
                            type_accuracy_scores.append(0.9)  # Correctly object
                    except:
                        type_accuracy_scores.append(0.9)  # Correctly object
                else:
                    type_accuracy_scores.append(0.8)
            else:
                # For non-object columns, assume correct type
                type_accuracy_scores.append(0.95)
        
        return np.mean(type_accuracy_scores) if type_accuracy_scores else 0.9
    
    def _calculate_format_accuracy(self, df: pd.DataFrame) -> float:
        """Calculate format accuracy."""
        format_accuracy_scores = []
        
        for col in df.select_dtypes(include=['object']).columns:
            non_null_values = df[col].dropna()
            if len(non_null_values) == 0:
                continue
            
            # Check for common format patterns
            if 'email' in col.lower():
                email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                valid_emails = non_null_values.str.match(email_pattern).mean()
                format_accuracy_scores.append(valid_emails)
            elif 'phone' in col.lower():
                phone_pattern = r'^[\+]?[1-9][\d]{0,15}$'
                valid_phones = non_null_values.str.match(phone_pattern).mean()
                format_accuracy_scores.append(valid_phones)
            elif 'date' in col.lower():
                try:
                    pd.to_datetime(non_null_values)
                    format_accuracy_scores.append(0.95)
                except:
                    format_accuracy_scores.append(0.7)
            else:
                # For other columns, assume good format
                format_accuracy_scores.append(0.9)
        
        return np.mean(format_accuracy_scores) if format_accuracy_scores else 0.9
    
    def _calculate_domain_accuracy(self, df: pd.DataFrame, context: QualityContext) -> float:
        """Calculate domain-specific accuracy."""
        # Domain-specific validation based on business context
        domain_rules = context.usage_patterns.get('domain_rules', {})
        
        if not domain_rules:
            return 0.85  # Default if no domain rules
        
        domain_accuracy_scores = []
        
        for col, rules in domain_rules.items():
            if col not in df.columns:
                continue
            
            col_values = df[col].dropna()
            if len(col_values) == 0:
                continue
            
            # Apply domain-specific rules
            if 'range' in rules:
                min_val, max_val = rules['range']
                if df[col].dtype in ['int64', 'float64']:
                    valid_range = ((col_values >= min_val) & (col_values <= max_val)).mean()
                    domain_accuracy_scores.append(valid_range)
            
            if 'valid_values' in rules:
                valid_values = rules['valid_values']
                valid_domain = col_values.isin(valid_values).mean()
                domain_accuracy_scores.append(valid_domain)
        
        return np.mean(domain_accuracy_scores) if domain_accuracy_scores else 0.85


class ConsistencyCalculator(QualityDimensionCalculator):
    """Calculator for consistency dimension."""
    
    def get_dimension(self) -> QualityDimension:
        return QualityDimension.CONSISTENCY
    
    def calculate(self, df: pd.DataFrame, 
                 validation_results: List[ValidationResult],
                 context: QualityContext) -> DimensionScore:
        """Calculate consistency score."""
        if len(df) == 0:
            return DimensionScore(
                dimension=QualityDimension.CONSISTENCY,
                score=0.0,
                weighted_score=0.0,
                weight=1.0,
                confidence=0.0,
                calculation_method="empty_dataset"
            )
        
        # Calculate format consistency
        format_consistency = self._calculate_format_consistency(df)
        
        # Calculate value consistency
        value_consistency = self._calculate_value_consistency(df)
        
        # Calculate cross-column consistency
        cross_column_consistency = self._calculate_cross_column_consistency(df, context)
        
        # Calculate temporal consistency
        temporal_consistency = self._calculate_temporal_consistency(df)
        
        # Combined consistency score
        consistencies = [format_consistency, value_consistency, cross_column_consistency, temporal_consistency]
        overall_consistency = np.mean(consistencies)
        
        # Calculate confidence
        confidence = 0.8
        
        # Generate recommendations
        recommendations = []
        if format_consistency < 0.9:
            recommendations.append("Standardize data formats across all records")
        if value_consistency < 0.9:
            recommendations.append("Implement value standardization rules")
        if cross_column_consistency < 0.9:
            recommendations.append("Add cross-column validation rules")
        
        return DimensionScore(
            dimension=QualityDimension.CONSISTENCY,
            score=overall_consistency,
            weighted_score=overall_consistency,
            weight=1.0,
            confidence=confidence,
            sub_scores={
                'format_consistency': format_consistency,
                'value_consistency': value_consistency,
                'cross_column_consistency': cross_column_consistency,
                'temporal_consistency': temporal_consistency
            },
            improvement_recommendations=recommendations,
            calculation_method="multi_aspect_analysis"
        )
    
    def _calculate_format_consistency(self, df: pd.DataFrame) -> float:
        """Calculate format consistency."""
        format_scores = []
        
        for col in df.select_dtypes(include=['object']).columns:
            non_null_values = df[col].dropna()
            if len(non_null_values) == 0:
                continue
            
            # Check format patterns
            if len(non_null_values) > 1:
                # Check length consistency
                lengths = non_null_values.str.len()
                length_consistency = 1 - (lengths.std() / lengths.mean()) if lengths.mean() > 0 else 0
                
                # Check character pattern consistency
                patterns = non_null_values.str.extract(r'([A-Za-z]+)|([0-9]+)|([^A-Za-z0-9]+)')
                pattern_consistency = 0.8  # Default
                
                format_scores.append(np.mean([length_consistency, pattern_consistency]))
            else:
                format_scores.append(1.0)
        
        return np.mean(format_scores) if format_scores else 0.9
    
    def _calculate_value_consistency(self, df: pd.DataFrame) -> float:
        """Calculate value consistency."""
        value_scores = []
        
        for col in df.columns:
            non_null_values = df[col].dropna()
            if len(non_null_values) == 0:
                continue
            
            if df[col].dtype in ['int64', 'float64']:
                # For numeric columns, check for consistent scale/range
                if len(non_null_values) > 1:
                    cv = non_null_values.std() / non_null_values.mean() if non_null_values.mean() > 0 else 0
                    # Lower coefficient of variation indicates more consistency
                    consistency_score = max(0, 1 - cv)
                    value_scores.append(consistency_score)
                else:
                    value_scores.append(1.0)
            else:
                # For categorical columns, check value distribution
                value_counts = non_null_values.value_counts()
                if len(value_counts) > 1:
                    # More even distribution indicates better consistency
                    entropy = -sum(p * np.log2(p) for p in value_counts / len(non_null_values) if p > 0)
                    max_entropy = np.log2(len(value_counts))
                    consistency_score = entropy / max_entropy if max_entropy > 0 else 0
                    value_scores.append(consistency_score)
                else:
                    value_scores.append(0.5)  # Single value - not necessarily good
        
        return np.mean(value_scores) if value_scores else 0.8
    
    def _calculate_cross_column_consistency(self, df: pd.DataFrame, context: QualityContext) -> float:
        """Calculate cross-column consistency."""
        consistency_rules = context.usage_patterns.get('consistency_rules', {})
        
        if not consistency_rules:
            return 0.85  # Default if no rules
        
        consistency_scores = []
        
        for rule_name, rule_config in consistency_rules.items():
            if 'columns' not in rule_config or len(rule_config['columns']) < 2:
                continue
            
            columns = rule_config['columns']
            if not all(col in df.columns for col in columns):
                continue
            
            # Check consistency between columns
            rule_type = rule_config.get('type', 'correlation')
            
            if rule_type == 'correlation':
                # Check correlation between numeric columns
                numeric_cols = [col for col in columns if df[col].dtype in ['int64', 'float64']]
                if len(numeric_cols) >= 2:
                    corr_matrix = df[numeric_cols].corr()
                    avg_correlation = corr_matrix.abs().mean().mean()
                    consistency_scores.append(avg_correlation)
            
            elif rule_type == 'dependency':
                # Check if one column depends on another
                col1, col2 = columns[:2]
                unique_combinations = df[[col1, col2]].drop_duplicates()
                dependency_ratio = 1 - (len(unique_combinations) / len(df)) if len(df) > 0 else 0
                consistency_scores.append(dependency_ratio)
        
        return np.mean(consistency_scores) if consistency_scores else 0.85
    
    def _calculate_temporal_consistency(self, df: pd.DataFrame) -> float:
        """Calculate temporal consistency."""
        # Check if there are datetime columns
        datetime_columns = df.select_dtypes(include=['datetime64']).columns
        
        if len(datetime_columns) == 0:
            return 0.9  # No temporal data to assess
        
        temporal_scores = []
        
        for col in datetime_columns:
            non_null_dates = df[col].dropna()
            if len(non_null_dates) < 2:
                continue
            
            # Check for temporal order consistency
            sorted_dates = non_null_dates.sort_values()
            time_diffs = sorted_dates.diff().dropna()
            
            if len(time_diffs) > 0:
                # Check consistency of time intervals
                mean_diff = time_diffs.mean()
                if mean_diff.total_seconds() > 0:
                    cv = time_diffs.std() / mean_diff
                    consistency_score = max(0, 1 - cv.total_seconds() / 86400)  # Normalize by day
                    temporal_scores.append(consistency_score)
        
        return np.mean(temporal_scores) if temporal_scores else 0.9


@dataclass(frozen=True)
class ComprehensiveQualityScoringConfig:
    """Configuration for comprehensive quality scoring."""
    # Scoring algorithm
    scoring_algorithm: ScoringAlgorithm = ScoringAlgorithm.WEIGHTED_AVERAGE
    
    # Quality weights
    quality_weights: List[QualityWeight] = field(default_factory=lambda: [
        QualityWeight(QualityDimension.COMPLETENESS, 0.2, critical_threshold=0.8),
        QualityWeight(QualityDimension.ACCURACY, 0.25, critical_threshold=0.7),
        QualityWeight(QualityDimension.CONSISTENCY, 0.15, critical_threshold=0.6),
        QualityWeight(QualityDimension.VALIDITY, 0.2, critical_threshold=0.8),
        QualityWeight(QualityDimension.UNIQUENESS, 0.1, critical_threshold=0.9),
        QualityWeight(QualityDimension.TIMELINESS, 0.1, critical_threshold=0.5)
    ])
    
    # Calculation options
    enable_benchmarking: bool = True
    enable_trend_analysis: bool = True
    enable_prediction: bool = True
    enable_business_impact: bool = True
    
    # Performance options
    enable_caching: bool = True
    cache_ttl_minutes: int = 60
    parallel_processing: bool = False
    
    # Reporting options
    detailed_reporting: bool = True
    include_recommendations: bool = True
    max_recommendations: int = 10


class ComprehensiveQualityScoringEngine:
    """Engine for comprehensive quality scoring."""
    
    def __init__(self, config: ComprehensiveQualityScoringConfig = None):
        """Initialize comprehensive quality scoring engine.
        
        Args:
            config: Service configuration
        """
        self.config = config or ComprehensiveQualityScoringConfig()
        
        # Initialize dimension calculators
        self._calculators = {
            QualityDimension.COMPLETENESS: CompletenessCalculator(),
            QualityDimension.ACCURACY: AccuracyCalculator(),
            QualityDimension.CONSISTENCY: ConsistencyCalculator(),
            # Add more calculators as needed
        }
        
        # Create weights lookup
        self._weights_lookup = {
            weight.dimension: weight for weight in self.config.quality_weights
        }
        
        # Cache for scores
        self._score_cache = {} if self.config.enable_caching else None
        
        logger.info("Comprehensive Quality Scoring Engine initialized")
    
    def calculate_comprehensive_score(self,
                                    df: pd.DataFrame,
                                    dataset_id: str,
                                    validation_results: List[ValidationResult] = None,
                                    quality_context: QualityContext = None) -> ComprehensiveQualityScore:
        """Calculate comprehensive quality score.
        
        Args:
            df: Input DataFrame
            dataset_id: Dataset identifier
            validation_results: Optional validation results
            quality_context: Optional quality context
            
        Returns:
            Comprehensive quality score
        """
        start_time = datetime.now()
        
        # Default parameters
        validation_results = validation_results or []
        quality_context = quality_context or QualityContext()
        
        # Calculate dimension scores
        dimension_scores = {}
        
        for dimension in QualityDimension:
            if dimension in self._calculators:
                calculator = self._calculators[dimension]
                score = calculator.calculate(df, validation_results, quality_context)
                
                # Apply weight
                weight_config = self._weights_lookup.get(dimension, QualityWeight(dimension, 0.1))
                score.weight = weight_config.weight
                score.weighted_score = score.score * score.weight
                
                dimension_scores[dimension] = score
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(dimension_scores)
        
        # Calculate weighted score
        weighted_score = sum(score.weighted_score for score in dimension_scores.values())
        
        # Determine quality level
        quality_level = self._determine_quality_level(overall_score)
        
        # Calculate overall confidence
        confidence = np.mean([score.confidence for score in dimension_scores.values()])
        
        # Identify critical dimensions
        critical_dimensions = [dim for dim, score in dimension_scores.items() if score.is_critical()]
        
        # Calculate business impact
        business_impact = self._calculate_business_impact(dimension_scores, quality_context)
        
        # Generate risk assessment
        risk_assessment = self._generate_risk_assessment(dimension_scores, quality_context)
        
        # Create comprehensive score
        calculation_duration = (datetime.now() - start_time).total_seconds() * 1000
        
        comprehensive_score = ComprehensiveQualityScore(
            dataset_id=dataset_id,
            overall_score=overall_score,
            weighted_score=weighted_score,
            quality_level=quality_level,
            confidence=confidence,
            dimension_scores=dimension_scores,
            scoring_algorithm=self.config.scoring_algorithm,
            quality_context=quality_context,
            weights_used={dim: score.weight for dim, score in dimension_scores.items()},
            critical_dimensions=critical_dimensions,
            business_impact_score=business_impact,
            risk_assessment=risk_assessment,
            calculation_duration_ms=calculation_duration
        )
        
        # Add improvement opportunities
        comprehensive_score.improvement_opportunities = comprehensive_score.get_top_improvement_opportunities()
        
        logger.info(f"Calculated comprehensive quality score: {overall_score:.3f} ({quality_level.value})")
        
        return comprehensive_score
    
    def _calculate_overall_score(self, dimension_scores: Dict[QualityDimension, DimensionScore]) -> float:
        """Calculate overall quality score."""
        if not dimension_scores:
            return 0.0
        
        if self.config.scoring_algorithm == ScoringAlgorithm.WEIGHTED_AVERAGE:
            total_weighted_score = sum(score.weighted_score for score in dimension_scores.values())
            total_weight = sum(score.weight for score in dimension_scores.values())
            return total_weighted_score / total_weight if total_weight > 0 else 0.0
        
        elif self.config.scoring_algorithm == ScoringAlgorithm.GEOMETRIC_MEAN:
            scores = [score.score for score in dimension_scores.values()]
            weights = [score.weight for score in dimension_scores.values()]
            
            # Weighted geometric mean
            weighted_product = 1.0
            total_weight = sum(weights)
            
            for score, weight in zip(scores, weights):
                weighted_product *= (score ** (weight / total_weight))
            
            return weighted_product
        
        elif self.config.scoring_algorithm == ScoringAlgorithm.HARMONIC_MEAN:
            scores = [score.score for score in dimension_scores.values()]
            weights = [score.weight for score in dimension_scores.values()]
            
            # Weighted harmonic mean
            numerator = sum(weights)
            denominator = sum(weight / max(score, 0.001) for score, weight in zip(scores, weights))
            
            return numerator / denominator if denominator > 0 else 0.0
        
        elif self.config.scoring_algorithm == ScoringAlgorithm.MINIMUM:
            # Conservative approach - minimum score
            return min(score.score for score in dimension_scores.values())
        
        else:
            # Default to weighted average
            return self._calculate_overall_score(dimension_scores)
    
    def _determine_quality_level(self, score: float) -> QualityLevel:
        """Determine quality level based on score."""
        if score >= 0.95:
            return QualityLevel.EXCELLENT
        elif score >= 0.90:
            return QualityLevel.VERY_GOOD
        elif score >= 0.80:
            return QualityLevel.GOOD
        elif score >= 0.70:
            return QualityLevel.SATISFACTORY
        elif score >= 0.60:
            return QualityLevel.NEEDS_IMPROVEMENT
        elif score >= 0.40:
            return QualityLevel.POOR
        elif score >= 0.20:
            return QualityLevel.VERY_POOR
        else:
            return QualityLevel.CRITICAL
    
    def _calculate_business_impact(self, 
                                 dimension_scores: Dict[QualityDimension, DimensionScore],
                                 context: QualityContext) -> float:
        """Calculate business impact score."""
        if not self.config.enable_business_impact:
            return 0.0
        
        # Business impact weights for different dimensions
        business_impact_weights = {
            QualityDimension.ACCURACY: 0.3,
            QualityDimension.COMPLETENESS: 0.25,
            QualityDimension.VALIDITY: 0.2,
            QualityDimension.TIMELINESS: 0.15,
            QualityDimension.CONSISTENCY: 0.1
        }
        
        impact_score = 0.0
        total_weight = 0.0
        
        for dimension, score in dimension_scores.items():
            if dimension in business_impact_weights:
                weight = business_impact_weights[dimension]
                # Lower quality has higher business impact (negative)
                impact_score += (1 - score.score) * weight
                total_weight += weight
        
        # Apply context sensitivity multiplier
        sensitivity_multiplier = context.get_sensitivity_multiplier()
        business_impact = (impact_score / total_weight) * sensitivity_multiplier if total_weight > 0 else 0.0
        
        return min(1.0, business_impact)
    
    def _generate_risk_assessment(self, 
                                dimension_scores: Dict[QualityDimension, DimensionScore],
                                context: QualityContext) -> Dict[str, Any]:
        """Generate risk assessment."""
        risk_levels = {
            QualityLevel.CRITICAL: "very_high",
            QualityLevel.VERY_POOR: "high",
            QualityLevel.POOR: "medium",
            QualityLevel.NEEDS_IMPROVEMENT: "low",
            QualityLevel.SATISFACTORY: "very_low",
            QualityLevel.GOOD: "very_low",
            QualityLevel.VERY_GOOD: "very_low",
            QualityLevel.EXCELLENT: "very_low"
        }
        
        critical_dimensions = [dim for dim, score in dimension_scores.items() if score.is_critical()]
        
        overall_risk = "very_high" if critical_dimensions else "low"
        
        return {
            'overall_risk': overall_risk,
            'critical_dimensions': [dim.value for dim in critical_dimensions],
            'risk_factors': {
                'data_sensitivity': context.data_sensitivity,
                'regulatory_requirements': len(context.regulatory_requirements),
                'business_domain': context.business_domain
            },
            'mitigation_recommendations': self._generate_mitigation_recommendations(critical_dimensions)
        }
    
    def _generate_mitigation_recommendations(self, critical_dimensions: List[QualityDimension]) -> List[str]:
        """Generate risk mitigation recommendations."""
        recommendations = []
        
        for dimension in critical_dimensions:
            if dimension == QualityDimension.COMPLETENESS:
                recommendations.append("Implement data collection improvements to address missing values")
            elif dimension == QualityDimension.ACCURACY:
                recommendations.append("Establish data validation rules and quality checks")
            elif dimension == QualityDimension.CONSISTENCY:
                recommendations.append("Standardize data formats and implement consistency rules")
            elif dimension == QualityDimension.VALIDITY:
                recommendations.append("Strengthen business rule validation and constraint enforcement")
            elif dimension == QualityDimension.UNIQUENESS:
                recommendations.append("Implement duplicate detection and prevention mechanisms")
            elif dimension == QualityDimension.TIMELINESS:
                recommendations.append("Optimize data refresh cycles and reduce processing latency")
        
        return recommendations
    
    def get_scoring_configuration(self) -> Dict[str, Any]:
        """Get current scoring configuration."""
        return {
            'scoring_algorithm': self.config.scoring_algorithm.value,
            'quality_weights': {
                weight.dimension.value: {
                    'weight': weight.weight,
                    'critical_threshold': weight.critical_threshold,
                    'min_threshold': weight.min_threshold,
                    'max_threshold': weight.max_threshold
                }
                for weight in self.config.quality_weights
            },
            'features_enabled': {
                'benchmarking': self.config.enable_benchmarking,
                'trend_analysis': self.config.enable_trend_analysis,
                'prediction': self.config.enable_prediction,
                'business_impact': self.config.enable_business_impact,
                'caching': self.config.enable_caching,
                'parallel_processing': self.config.parallel_processing
            },
            'reporting_options': {
                'detailed_reporting': self.config.detailed_reporting,
                'include_recommendations': self.config.include_recommendations,
                'max_recommendations': self.config.max_recommendations
            }
        }