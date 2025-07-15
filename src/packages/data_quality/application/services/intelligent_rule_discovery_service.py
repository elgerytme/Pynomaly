"""Intelligent Quality Rule Discovery Service.

Service for automatically discovering validation rules from data patterns,
statistical analysis, and machine learning techniques.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging
from collections import defaultdict, Counter
from enum import Enum
import re
from statistics import mean, median, stdev
import warnings

# ML imports for rule discovery
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif, chi2, SelectKBest
from sklearn.model_selection import train_test_split
from scipy import stats
from scipy.stats import kstest, normaltest, jarque_bera

from ...domain.entities.validation_rule import (
    QualityRule, ValidationLogic, RuleId, LogicType, Severity,
    SuccessCriteria, RuleCategory, RuleScope
)
from ...domain.entities.quality_profile import DatasetId
from .quality_assessment_service import QualityAssessmentService

logger = logging.getLogger(__name__)


class RuleDiscoveryMethod(Enum):
    """Rule discovery methods."""
    STATISTICAL_ANALYSIS = "statistical_analysis"
    PATTERN_RECOGNITION = "pattern_recognition"
    ASSOCIATION_RULES = "association_rules"
    DECISION_TREE_RULES = "decision_tree_rules"
    CLUSTERING_RULES = "clustering_rules"
    CORRELATION_RULES = "correlation_rules"
    DOMAIN_SPECIFIC = "domain_specific"
    OUTLIER_DETECTION = "outlier_detection"
    DISTRIBUTION_FITTING = "distribution_fitting"
    BUSINESS_LOGIC_INFERENCE = "business_logic_inference"


class RuleConfidence(Enum):
    """Confidence levels for discovered rules."""
    VERY_HIGH = "very_high"  # >95%
    HIGH = "high"           # 85-95%
    MEDIUM = "medium"       # 70-85%
    LOW = "low"            # 50-70%
    VERY_LOW = "very_low"  # <50%


@dataclass(frozen=True)
class RuleDiscoveryConfig:
    """Configuration for intelligent rule discovery."""
    # Discovery methods to use
    enabled_methods: List[RuleDiscoveryMethod] = field(default_factory=lambda: [
        RuleDiscoveryMethod.STATISTICAL_ANALYSIS,
        RuleDiscoveryMethod.PATTERN_RECOGNITION,
        RuleDiscoveryMethod.CORRELATION_RULES,
        RuleDiscoveryMethod.OUTLIER_DETECTION
    ])
    
    # Statistical thresholds
    min_confidence: float = 0.8
    min_support: float = 0.1
    max_rules_per_method: int = 50
    
    # Pattern recognition
    pattern_min_frequency: int = 10
    regex_confidence_threshold: float = 0.9
    
    # Statistical analysis
    normality_test_alpha: float = 0.05
    outlier_threshold: float = 3.0
    correlation_threshold: float = 0.7
    
    # Business logic inference
    enable_domain_rules: bool = True
    common_domains: List[str] = field(default_factory=lambda: [
        'email', 'phone', 'ssn', 'date', 'currency', 'url', 'ip_address'
    ])
    
    # Rule optimization
    enable_rule_ranking: bool = True
    enable_rule_merging: bool = True
    enable_redundancy_removal: bool = True
    
    # Performance
    max_sample_size: int = 100000
    enable_parallel_discovery: bool = False
    
    def __post_init__(self):
        """Validate configuration."""
        if not 0 < self.min_confidence <= 1:
            raise ValueError("Min confidence must be between 0 and 1")
        if not 0 < self.min_support <= 1:
            raise ValueError("Min support must be between 0 and 1")


@dataclass
class DiscoveredRule:
    """Container for a discovered quality rule."""
    rule: QualityRule
    discovery_method: RuleDiscoveryMethod
    confidence: float
    support: float
    evidence: Dict[str, Any]
    rule_explanation: str
    effectiveness_score: float = 0.0
    
    def get_confidence_level(self) -> RuleConfidence:
        """Get confidence level."""
        if self.confidence >= 0.95:
            return RuleConfidence.VERY_HIGH
        elif self.confidence >= 0.85:
            return RuleConfidence.HIGH
        elif self.confidence >= 0.70:
            return RuleConfidence.MEDIUM
        elif self.confidence >= 0.50:
            return RuleConfidence.LOW
        else:
            return RuleConfidence.VERY_LOW
    
    def is_high_quality(self) -> bool:
        """Check if rule is high quality."""
        return (self.confidence >= 0.8 and 
                self.support >= 0.1 and 
                self.effectiveness_score >= 0.7)


class IntelligentRuleDiscoveryService:
    """Service for intelligent quality rule discovery."""
    
    def __init__(self, config: RuleDiscoveryConfig = None):
        """Initialize rule discovery service."""
        self.config = config or RuleDiscoveryConfig()
        self.quality_assessment_service = QualityAssessmentService()
        
        # Rule templates and patterns
        self._rule_templates = self._initialize_rule_templates()
        self._domain_patterns = self._initialize_domain_patterns()
        
        # Discovery statistics
        self._discovery_stats = {
            'total_discoveries': 0,
            'rules_discovered': 0,
            'high_quality_rules': 0,
            'rules_by_method': defaultdict(int),
            'avg_confidence': 0.0,
            'avg_support': 0.0
        }
    
    def discover_quality_rules(self, 
                             df: pd.DataFrame,
                             dataset_id: str,
                             existing_rules: List[QualityRule] = None,
                             domain_context: Dict[str, Any] = None) -> List[DiscoveredRule]:
        """Discover quality rules from dataset."""
        try:
            logger.info(f"Starting intelligent rule discovery for dataset {dataset_id}")
            
            # Apply sampling if dataset is large
            if len(df) > self.config.max_sample_size:
                df_sample = df.sample(n=self.config.max_sample_size, random_state=42)
                logger.info(f"Applied sampling: {len(df_sample)} rows")
            else:
                df_sample = df
            
            discovered_rules = []
            
            # Run discovery with each enabled method
            for method in self.config.enabled_methods:
                try:
                    method_rules = self._discover_with_method(
                        df_sample, method, dataset_id, domain_context
                    )
                    discovered_rules.extend(method_rules)
                    self._discovery_stats['rules_by_method'][method.value] += len(method_rules)
                    logger.info(f"Method {method.value} discovered {len(method_rules)} rules")
                except Exception as e:
                    logger.error(f"Rule discovery with method {method.value} failed: {str(e)}")
            
            # Filter and optimize rules
            if discovered_rules:
                # Remove duplicates
                if self.config.enable_redundancy_removal:
                    discovered_rules = self._remove_redundant_rules(discovered_rules)
                
                # Merge similar rules
                if self.config.enable_rule_merging:
                    discovered_rules = self._merge_similar_rules(discovered_rules)
                
                # Rank rules
                if self.config.enable_rule_ranking:
                    discovered_rules = self._rank_rules(discovered_rules)
                
                # Filter by confidence
                discovered_rules = [
                    rule for rule in discovered_rules
                    if rule.confidence >= self.config.min_confidence
                ]
            
            # Update statistics
            self._update_discovery_stats(discovered_rules)
            
            logger.info(f"Discovered {len(discovered_rules)} quality rules")
            return discovered_rules
            
        except Exception as e:
            logger.error(f"Rule discovery failed: {str(e)}")
            raise
    
    def suggest_rule_improvements(self, 
                                existing_rules: List[QualityRule],
                                validation_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Suggest improvements for existing rules."""
        try:
            suggestions = []
            
            for rule in existing_rules:
                # Find validation results for this rule
                rule_results = [
                    result for result in validation_results
                    if result.get('rule_id') == str(rule.rule_id)
                ]
                
                if not rule_results:
                    continue
                
                # Analyze rule performance
                total_validations = len(rule_results)
                failed_validations = sum(
                    1 for result in rule_results
                    if result.get('status') in ['failed', 'error']
                )
                false_positive_rate = failed_validations / total_validations if total_validations > 0 else 0
                
                # Generate suggestions
                rule_suggestions = []
                
                # High false positive rate
                if false_positive_rate > 0.2:
                    rule_suggestions.append({
                        'type': 'threshold_adjustment',
                        'description': f'High false positive rate ({false_positive_rate:.1%}). Consider relaxing thresholds.',
                        'priority': 'high'
                    })
                
                # Low effectiveness
                avg_failure_rate = mean([
                    result.get('failure_rate', 0) for result in rule_results
                ])
                if avg_failure_rate < 0.05:
                    rule_suggestions.append({
                        'type': 'threshold_tightening',
                        'description': f'Low failure detection rate ({avg_failure_rate:.1%}). Consider tightening thresholds.',
                        'priority': 'medium'
                    })
                
                # Performance issues
                avg_execution_time = mean([
                    result.get('execution_time_ms', 0) for result in rule_results
                ])
                if avg_execution_time > 1000:  # > 1 second
                    rule_suggestions.append({
                        'type': 'performance_optimization',
                        'description': f'High execution time ({avg_execution_time:.0f}ms). Consider rule optimization.',
                        'priority': 'medium'
                    })
                
                if rule_suggestions:
                    suggestions.append({
                        'rule_id': str(rule.rule_id),
                        'rule_name': rule.rule_name,
                        'suggestions': rule_suggestions,
                        'performance_summary': {
                            'total_validations': total_validations,
                            'false_positive_rate': false_positive_rate,
                            'avg_failure_rate': avg_failure_rate,
                            'avg_execution_time_ms': avg_execution_time
                        }
                    })
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Rule improvement suggestion failed: {str(e)}")
            raise
    
    def analyze_rule_effectiveness(self, 
                                 rules: List[QualityRule],
                                 df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze effectiveness of quality rules."""
        try:
            effectiveness_analysis = {
                'total_rules': len(rules),
                'rule_analysis': [],
                'overall_coverage': 0.0,
                'redundancy_score': 0.0,
                'completeness_score': 0.0
            }
            
            for rule in rules:
                # Simulate rule execution (simplified)
                rule_analysis = {
                    'rule_id': str(rule.rule_id),
                    'rule_name': rule.rule_name,
                    'category': rule.category.value if rule.category else 'unknown',
                    'estimated_coverage': self._estimate_rule_coverage(rule, df),
                    'complexity_score': self._calculate_rule_complexity(rule),
                    'maintainability_score': self._calculate_maintainability(rule),
                    'business_value_score': self._estimate_business_value(rule),
                    'recommendations': []
                }
                
                # Generate recommendations
                if rule_analysis['complexity_score'] > 0.8:
                    rule_analysis['recommendations'].append(
                        'Consider simplifying rule logic for better maintainability'
                    )
                
                if rule_analysis['estimated_coverage'] < 0.1:
                    rule_analysis['recommendations'].append(
                        'Rule has low coverage - consider broadening scope or removing'
                    )
                
                effectiveness_analysis['rule_analysis'].append(rule_analysis)
            
            # Calculate overall metrics
            if effectiveness_analysis['rule_analysis']:
                effectiveness_analysis['overall_coverage'] = mean([
                    analysis['estimated_coverage'] 
                    for analysis in effectiveness_analysis['rule_analysis']
                ])
                
                effectiveness_analysis['avg_complexity'] = mean([
                    analysis['complexity_score'] 
                    for analysis in effectiveness_analysis['rule_analysis']
                ])
                
                effectiveness_analysis['avg_business_value'] = mean([
                    analysis['business_value_score'] 
                    for analysis in effectiveness_analysis['rule_analysis']
                ])
            
            return effectiveness_analysis
            
        except Exception as e:
            logger.error(f"Rule effectiveness analysis failed: {str(e)}")
            raise
    
    def _discover_with_method(self, 
                            df: pd.DataFrame,
                            method: RuleDiscoveryMethod,
                            dataset_id: str,
                            domain_context: Dict[str, Any] = None) -> List[DiscoveredRule]:
        """Discover rules using specific method."""
        try:
            if method == RuleDiscoveryMethod.STATISTICAL_ANALYSIS:
                return self._discover_statistical_rules(df, dataset_id)
            elif method == RuleDiscoveryMethod.PATTERN_RECOGNITION:
                return self._discover_pattern_rules(df, dataset_id)
            elif method == RuleDiscoveryMethod.CORRELATION_RULES:
                return self._discover_correlation_rules(df, dataset_id)
            elif method == RuleDiscoveryMethod.OUTLIER_DETECTION:
                return self._discover_outlier_rules(df, dataset_id)
            elif method == RuleDiscoveryMethod.DECISION_TREE_RULES:
                return self._discover_decision_tree_rules(df, dataset_id)
            elif method == RuleDiscoveryMethod.CLUSTERING_RULES:
                return self._discover_clustering_rules(df, dataset_id)
            elif method == RuleDiscoveryMethod.DOMAIN_SPECIFIC:
                return self._discover_domain_specific_rules(df, dataset_id, domain_context)
            elif method == RuleDiscoveryMethod.DISTRIBUTION_FITTING:
                return self._discover_distribution_rules(df, dataset_id)
            else:
                logger.warning(f"Unknown discovery method: {method}")
                return []
                
        except Exception as e:
            logger.error(f"Discovery method {method.value} failed: {str(e)}")
            return []
    
    def _discover_statistical_rules(self, df: pd.DataFrame, dataset_id: str) -> List[DiscoveredRule]:
        """Discover rules using statistical analysis."""
        discovered_rules = []
        
        for column in df.columns:
            if df[column].dtype in ['int64', 'float64']:
                # Numerical column rules
                
                # Range validation rule
                q1 = df[column].quantile(0.25)
                q3 = df[column].quantile(0.75)
                iqr = q3 - q1
                
                if iqr > 0:
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    # Create range validation rule
                    rule = QualityRule(
                        rule_id=RuleId(),
                        rule_name=f"{column}_range_validation",
                        description=f"Values in {column} should be within statistical range",
                        category=RuleCategory.RANGE_VALIDATION,
                        scope=RuleScope.COLUMN,
                        target_columns=[column],
                        validation_logic=ValidationLogic(
                            logic_type=LogicType.STATISTICAL,
                            expression=f"{column} >= {lower_bound} AND {column} <= {upper_bound}",
                            parameters={
                                'column_name': column,
                                'stat_type': 'range',
                                'min_value': lower_bound,
                                'max_value': upper_bound
                            },
                            success_criteria=SuccessCriteria(min_pass_rate=0.95),
                            error_message=f"Value outside valid range [{lower_bound:.2f}, {upper_bound:.2f}]"
                        ),
                        severity=Severity.MEDIUM,
                        is_active=True
                    )
                    
                    # Calculate confidence and support
                    values_in_range = ((df[column] >= lower_bound) & (df[column] <= upper_bound)).sum()
                    confidence = values_in_range / len(df)
                    support = len(df[column].dropna()) / len(df)
                    
                    discovered_rules.append(DiscoveredRule(
                        rule=rule,
                        discovery_method=RuleDiscoveryMethod.STATISTICAL_ANALYSIS,
                        confidence=confidence,
                        support=support,
                        evidence={
                            'q1': q1,
                            'q3': q3,
                            'iqr': iqr,
                            'outliers_detected': len(df) - values_in_range
                        },
                        rule_explanation=f"Statistical analysis detected outliers using IQR method",
                        effectiveness_score=min(1.0, confidence * support * 2)
                    ))
                
                # Normality test rule
                if len(df[column].dropna()) > 8:  # Minimum for normality test
                    try:
                        statistic, p_value = normaltest(df[column].dropna())
                        is_normal = p_value > self.config.normality_test_alpha
                        
                        if is_normal:
                            mean_val = df[column].mean()
                            std_val = df[column].std()
                            
                            # Create normality-based validation rule
                            rule = QualityRule(
                                rule_id=RuleId(),
                                rule_name=f"{column}_normality_validation",
                                description=f"Values in {column} should follow normal distribution pattern",
                                category=RuleCategory.STATISTICAL_VALIDATION,
                                scope=RuleScope.COLUMN,
                                target_columns=[column],
                                validation_logic=ValidationLogic(
                                    logic_type=LogicType.STATISTICAL,
                                    expression=f"ABS(({column} - {mean_val}) / {std_val}) <= 3",
                                    parameters={
                                        'column_name': column,
                                        'stat_type': 'z_score',
                                        'threshold': 3.0,
                                        'mean': mean_val,
                                        'std': std_val
                                    },
                                    success_criteria=SuccessCriteria(min_pass_rate=0.99),
                                    error_message=f"Value deviates significantly from normal distribution"
                                ),
                                severity=Severity.LOW,
                                is_active=True
                            )
                            
                            confidence = min(1.0, 1 - p_value)
                            support = len(df[column].dropna()) / len(df)
                            
                            discovered_rules.append(DiscoveredRule(
                                rule=rule,
                                discovery_method=RuleDiscoveryMethod.STATISTICAL_ANALYSIS,
                                confidence=confidence,
                                support=support,
                                evidence={
                                    'normality_statistic': statistic,
                                    'p_value': p_value,
                                    'mean': mean_val,
                                    'std': std_val
                                },
                                rule_explanation=f"Data follows normal distribution (p-value: {p_value:.4f})",
                                effectiveness_score=confidence * support
                            ))
                    except Exception:
                        pass  # Skip normality test if it fails
            
            else:
                # Categorical column rules
                
                # Completeness rule
                null_count = df[column].isnull().sum()
                completeness_rate = 1 - (null_count / len(df))
                
                if completeness_rate > 0.9:  # High completeness
                    rule = QualityRule(
                        rule_id=RuleId(),
                        rule_name=f"{column}_completeness_validation",
                        description=f"Column {column} should have high completeness",
                        category=RuleCategory.COMPLETENESS_VALIDATION,
                        scope=RuleScope.COLUMN,
                        target_columns=[column],
                        validation_logic=ValidationLogic(
                            logic_type=LogicType.PYTHON,
                            expression=f"df['{column}'].notna()",
                            parameters={'column_name': column},
                            success_criteria=SuccessCriteria(min_pass_rate=completeness_rate * 0.95),
                            error_message=f"Missing value in {column}"
                        ),
                        severity=Severity.MEDIUM,
                        is_active=True
                    )
                    
                    discovered_rules.append(DiscoveredRule(
                        rule=rule,
                        discovery_method=RuleDiscoveryMethod.STATISTICAL_ANALYSIS,
                        confidence=completeness_rate,
                        support=1.0,
                        evidence={
                            'null_count': null_count,
                            'total_count': len(df),
                            'completeness_rate': completeness_rate
                        },
                        rule_explanation=f"Column has high completeness rate ({completeness_rate:.1%})",
                        effectiveness_score=completeness_rate
                    ))
                
                # Unique value count rule
                unique_count = df[column].nunique()
                unique_ratio = unique_count / len(df)
                
                if unique_ratio > 0.95:  # Likely unique identifier
                    rule = QualityRule(
                        rule_id=RuleId(),
                        rule_name=f"{column}_uniqueness_validation",
                        description=f"Column {column} should have unique values",
                        category=RuleCategory.UNIQUENESS_VALIDATION,
                        scope=RuleScope.COLUMN,
                        target_columns=[column],
                        validation_logic=ValidationLogic(
                            logic_type=LogicType.PYTHON,
                            expression=f"~df['{column}'].duplicated()",
                            parameters={'column_name': column},
                            success_criteria=SuccessCriteria(min_pass_rate=0.99),
                            error_message=f"Duplicate value in {column}"
                        ),
                        severity=Severity.HIGH,
                        is_active=True
                    )
                    
                    discovered_rules.append(DiscoveredRule(
                        rule=rule,
                        discovery_method=RuleDiscoveryMethod.STATISTICAL_ANALYSIS,
                        confidence=unique_ratio,
                        support=1.0,
                        evidence={
                            'unique_count': unique_count,
                            'total_count': len(df),
                            'unique_ratio': unique_ratio
                        },
                        rule_explanation=f"Column has high uniqueness ({unique_ratio:.1%})",
                        effectiveness_score=unique_ratio
                    ))
        
        return discovered_rules[:self.config.max_rules_per_method]
    
    def _discover_pattern_rules(self, df: pd.DataFrame, dataset_id: str) -> List[DiscoveredRule]:
        """Discover rules using pattern recognition."""
        discovered_rules = []
        
        for column in df.select_dtypes(include=['object']).columns:
            # Extract patterns from string values
            values = df[column].dropna().astype(str)
            
            if len(values) == 0:
                continue
            
            # Analyze string patterns
            pattern_analysis = self._analyze_string_patterns(values)
            
            for pattern_info in pattern_analysis:
                if (pattern_info['frequency'] >= self.config.pattern_min_frequency and
                    pattern_info['confidence'] >= self.config.regex_confidence_threshold):
                    
                    # Create pattern validation rule
                    rule = QualityRule(
                        rule_id=RuleId(),
                        rule_name=f"{column}_pattern_validation",
                        description=f"Values in {column} should match pattern: {pattern_info['description']}",
                        category=RuleCategory.FORMAT_VALIDATION,
                        scope=RuleScope.COLUMN,
                        target_columns=[column],
                        validation_logic=ValidationLogic(
                            logic_type=LogicType.REGEX,
                            expression=pattern_info['regex'],
                            parameters={
                                'column_name': column,
                                'pattern': pattern_info['regex']
                            },
                            success_criteria=SuccessCriteria(min_pass_rate=pattern_info['confidence']),
                            error_message=f"Value does not match expected pattern: {pattern_info['description']}"
                        ),
                        severity=Severity.MEDIUM,
                        is_active=True
                    )
                    
                    discovered_rules.append(DiscoveredRule(
                        rule=rule,
                        discovery_method=RuleDiscoveryMethod.PATTERN_RECOGNITION,
                        confidence=pattern_info['confidence'],
                        support=pattern_info['frequency'] / len(values),
                        evidence=pattern_info,
                        rule_explanation=f"Pattern discovered: {pattern_info['description']} (confidence: {pattern_info['confidence']:.1%})",
                        effectiveness_score=pattern_info['confidence'] * (pattern_info['frequency'] / len(values))
                    ))
        
        return discovered_rules[:self.config.max_rules_per_method]
    
    def _discover_correlation_rules(self, df: pd.DataFrame, dataset_id: str) -> List[DiscoveredRule]:
        """Discover rules based on column correlations."""
        discovered_rules = []
        
        # Select numerical columns for correlation analysis
        numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
        
        if len(numerical_columns) < 2:
            return []
        
        # Calculate correlation matrix
        correlation_matrix = df[numerical_columns].corr()
        
        # Find high correlations
        for i, col1 in enumerate(numerical_columns):
            for j, col2 in enumerate(numerical_columns):
                if i < j:  # Avoid duplicates
                    correlation = correlation_matrix.loc[col1, col2]
                    
                    if abs(correlation) >= self.config.correlation_threshold:
                        # Create correlation rule
                        if correlation > 0:
                            rule_description = f"{col1} and {col2} should be positively correlated"
                            expression = f"({col1} > {col1}.median()) == ({col2} > {col2}.median())"
                        else:
                            rule_description = f"{col1} and {col2} should be negatively correlated"
                            expression = f"({col1} > {col1}.median()) != ({col2} > {col2}.median())"
                        
                        rule = QualityRule(
                            rule_id=RuleId(),
                            rule_name=f"{col1}_{col2}_correlation_validation",
                            description=rule_description,
                            category=RuleCategory.CROSS_FIELD_VALIDATION,
                            scope=RuleScope.CROSS_COLUMN,
                            target_columns=[col1, col2],
                            validation_logic=ValidationLogic(
                                logic_type=LogicType.COMPARISON,
                                expression=expression,
                                parameters={
                                    'column1': col1,
                                    'column2': col2,
                                    'correlation': correlation,
                                    'operator': '==' if correlation > 0 else '!='
                                },
                                success_criteria=SuccessCriteria(min_pass_rate=0.8),
                                error_message=f"Correlation violation between {col1} and {col2}"
                            ),
                            severity=Severity.LOW,
                            is_active=True
                        )
                        
                        confidence = min(1.0, abs(correlation))
                        support = 1.0
                        
                        discovered_rules.append(DiscoveredRule(
                            rule=rule,
                            discovery_method=RuleDiscoveryMethod.CORRELATION_RULES,
                            confidence=confidence,
                            support=support,
                            evidence={
                                'correlation': correlation,
                                'correlation_type': 'positive' if correlation > 0 else 'negative',
                                'column1': col1,
                                'column2': col2
                            },
                            rule_explanation=f"Strong correlation detected: {correlation:.3f}",
                            effectiveness_score=confidence * 0.8  # Lower weight for correlation rules
                        ))
        
        return discovered_rules[:self.config.max_rules_per_method]
    
    def _discover_outlier_rules(self, df: pd.DataFrame, dataset_id: str) -> List[DiscoveredRule]:
        """Discover rules for outlier detection."""
        discovered_rules = []
        
        for column in df.select_dtypes(include=['int64', 'float64']).columns:
            values = df[column].dropna()
            
            if len(values) < 10:
                continue
            
            # Z-score based outlier detection
            mean_val = values.mean()
            std_val = values.std()
            
            if std_val > 0:
                z_scores = np.abs((values - mean_val) / std_val)
                outliers = z_scores > self.config.outlier_threshold
                outlier_rate = outliers.sum() / len(values)
                
                if outlier_rate < 0.1:  # Less than 10% outliers
                    rule = QualityRule(
                        rule_id=RuleId(),
                        rule_name=f"{column}_outlier_detection",
                        description=f"Values in {column} should not be statistical outliers",
                        category=RuleCategory.OUTLIER_DETECTION,
                        scope=RuleScope.COLUMN,
                        target_columns=[column],
                        validation_logic=ValidationLogic(
                            logic_type=LogicType.STATISTICAL,
                            expression=f"ABS(({column} - {mean_val}) / {std_val}) <= {self.config.outlier_threshold}",
                            parameters={
                                'column_name': column,
                                'stat_type': 'z_score',
                                'threshold': self.config.outlier_threshold,
                                'mean': mean_val,
                                'std': std_val
                            },
                            success_criteria=SuccessCriteria(min_pass_rate=1 - outlier_rate),
                            error_message=f"Statistical outlier detected in {column}"
                        ),
                        severity=Severity.MEDIUM,
                        is_active=True
                    )
                    
                    confidence = 1 - outlier_rate
                    support = len(values) / len(df)
                    
                    discovered_rules.append(DiscoveredRule(
                        rule=rule,
                        discovery_method=RuleDiscoveryMethod.OUTLIER_DETECTION,
                        confidence=confidence,
                        support=support,
                        evidence={
                            'mean': mean_val,
                            'std': std_val,
                            'outlier_threshold': self.config.outlier_threshold,
                            'outlier_count': outliers.sum(),
                            'outlier_rate': outlier_rate
                        },
                        rule_explanation=f"Z-score outlier detection (threshold: {self.config.outlier_threshold})",
                        effectiveness_score=confidence * support
                    ))
        
        return discovered_rules[:self.config.max_rules_per_method]
    
    def _discover_decision_tree_rules(self, df: pd.DataFrame, dataset_id: str) -> List[DiscoveredRule]:
        """Discover rules using decision tree analysis."""
        discovered_rules = []
        
        # This is a simplified implementation
        # In practice, you'd need labeled data or create synthetic labels based on quality issues
        
        try:
            # Create synthetic labels based on data quality issues
            labels = []
            for _, row in df.iterrows():
                # Simple quality scoring
                quality_score = 1.0
                
                # Check for missing values
                if row.isnull().any():
                    quality_score -= 0.3
                
                # Check for duplicates (simplified)
                if any(str(val).strip().lower() in ['duplicate', 'dup', 'copy'] for val in row.astype(str)):
                    quality_score -= 0.5
                
                labels.append(1 if quality_score > 0.7 else 0)
            
            # Prepare features
            features = []
            feature_names = []
            
            for column in df.columns:
                if df[column].dtype in ['int64', 'float64']:
                    features.append(df[column].fillna(0).values)
                    feature_names.append(column)
                else:
                    # Encode categorical variables
                    le = LabelEncoder()
                    encoded = le.fit_transform(df[column].astype(str))
                    features.append(encoded)
                    feature_names.append(column)
            
            if len(features) == 0:
                return []
            
            X = np.column_stack(features)
            y = np.array(labels)
            
            # Train decision tree
            dt = DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42)
            dt.fit(X, y)
            
            # Extract rules from decision tree
            tree_rules = export_text(dt, feature_names=feature_names)
            
            # Parse and convert to quality rules (simplified)
            # This would need more sophisticated parsing in practice
            
        except Exception as e:
            logger.error(f"Decision tree rule discovery failed: {str(e)}")
        
        return discovered_rules
    
    def _discover_clustering_rules(self, df: pd.DataFrame, dataset_id: str) -> List[DiscoveredRule]:
        """Discover rules using clustering analysis."""
        discovered_rules = []
        
        try:
            # Prepare numerical data for clustering
            numerical_data = df.select_dtypes(include=['int64', 'float64'])
            
            if len(numerical_data.columns) < 2:
                return []
            
            # Fill missing values
            data_filled = numerical_data.fillna(numerical_data.mean())
            
            # Perform clustering
            n_clusters = min(5, max(2, len(data_filled) // 100))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(data_filled)
            
            # Analyze clusters for rule discovery
            for cluster_id in range(n_clusters):
                cluster_mask = clusters == cluster_id
                cluster_data = data_filled[cluster_mask]
                
                if len(cluster_data) > 10:  # Sufficient cluster size
                    # Find cluster characteristics
                    cluster_means = cluster_data.mean()
                    cluster_stds = cluster_data.std()
                    
                    # Create cluster-based validation rules (simplified)
                    for column in cluster_data.columns:
                        if cluster_stds[column] > 0:
                            mean_val = cluster_means[column]
                            std_val = cluster_stds[column]
                            
                            # This would need more sophisticated logic in practice
                            # For now, skip cluster rules
                            pass
            
        except Exception as e:
            logger.error(f"Clustering rule discovery failed: {str(e)}")
        
        return discovered_rules
    
    def _discover_domain_specific_rules(self, 
                                      df: pd.DataFrame,
                                      dataset_id: str,
                                      domain_context: Dict[str, Any] = None) -> List[DiscoveredRule]:
        """Discover domain-specific validation rules."""
        discovered_rules = []
        
        if not self.config.enable_domain_rules:
            return []
        
        for column in df.select_dtypes(include=['object']).columns:
            column_lower = column.lower()
            sample_values = df[column].dropna().astype(str).str.lower()
            
            # Email validation
            if any(keyword in column_lower for keyword in ['email', 'mail']):
                email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                valid_emails = sample_values.str.match(email_pattern).sum()
                confidence = valid_emails / len(sample_values) if len(sample_values) > 0 else 0
                
                if confidence > 0.8:
                    rule = QualityRule(
                        rule_id=RuleId(),
                        rule_name=f"{column}_email_validation",
                        description=f"Values in {column} should be valid email addresses",
                        category=RuleCategory.FORMAT_VALIDATION,
                        scope=RuleScope.COLUMN,
                        target_columns=[column],
                        validation_logic=ValidationLogic(
                            logic_type=LogicType.REGEX,
                            expression=email_pattern,
                            parameters={'column_name': column, 'pattern': email_pattern},
                            success_criteria=SuccessCriteria(min_pass_rate=0.95),
                            error_message=f"Invalid email format"
                        ),
                        severity=Severity.HIGH,
                        is_active=True
                    )
                    
                    discovered_rules.append(DiscoveredRule(
                        rule=rule,
                        discovery_method=RuleDiscoveryMethod.DOMAIN_SPECIFIC,
                        confidence=confidence,
                        support=len(sample_values) / len(df),
                        evidence={
                            'domain': 'email',
                            'pattern': email_pattern,
                            'valid_count': valid_emails,
                            'total_count': len(sample_values)
                        },
                        rule_explanation="Email format validation based on column name pattern",
                        effectiveness_score=confidence
                    ))
            
            # Phone number validation
            elif any(keyword in column_lower for keyword in ['phone', 'tel', 'mobile']):
                phone_pattern = r'^[\d\s\-\(\)\+]{10,15}$'
                valid_phones = sample_values.str.match(phone_pattern).sum()
                confidence = valid_phones / len(sample_values) if len(sample_values) > 0 else 0
                
                if confidence > 0.7:
                    rule = QualityRule(
                        rule_id=RuleId(),
                        rule_name=f"{column}_phone_validation",
                        description=f"Values in {column} should be valid phone numbers",
                        category=RuleCategory.FORMAT_VALIDATION,
                        scope=RuleScope.COLUMN,
                        target_columns=[column],
                        validation_logic=ValidationLogic(
                            logic_type=LogicType.REGEX,
                            expression=phone_pattern,
                            parameters={'column_name': column, 'pattern': phone_pattern},
                            success_criteria=SuccessCriteria(min_pass_rate=0.9),
                            error_message=f"Invalid phone number format"
                        ),
                        severity=Severity.MEDIUM,
                        is_active=True
                    )
                    
                    discovered_rules.append(DiscoveredRule(
                        rule=rule,
                        discovery_method=RuleDiscoveryMethod.DOMAIN_SPECIFIC,
                        confidence=confidence,
                        support=len(sample_values) / len(df),
                        evidence={
                            'domain': 'phone',
                            'pattern': phone_pattern,
                            'valid_count': valid_phones,
                            'total_count': len(sample_values)
                        },
                        rule_explanation="Phone number format validation based on column name pattern",
                        effectiveness_score=confidence
                    ))
            
            # Date validation
            elif any(keyword in column_lower for keyword in ['date', 'time', 'created', 'updated']):
                # Try to parse as dates
                try:
                    parsed_dates = pd.to_datetime(sample_values, errors='coerce')
                    valid_dates = parsed_dates.notna().sum()
                    confidence = valid_dates / len(sample_values) if len(sample_values) > 0 else 0
                    
                    if confidence > 0.8:
                        rule = QualityRule(
                            rule_id=RuleId(),
                            rule_name=f"{column}_date_validation",
                            description=f"Values in {column} should be valid dates",
                            category=RuleCategory.FORMAT_VALIDATION,
                            scope=RuleScope.COLUMN,
                            target_columns=[column],
                            validation_logic=ValidationLogic(
                                logic_type=LogicType.PYTHON,
                                expression=f"pd.to_datetime(df['{column}'], errors='coerce').notna()",
                                parameters={'column_name': column},
                                success_criteria=SuccessCriteria(min_pass_rate=0.95),
                                error_message=f"Invalid date format"
                            ),
                            severity=Severity.MEDIUM,
                            is_active=True
                        )
                        
                        discovered_rules.append(DiscoveredRule(
                            rule=rule,
                            discovery_method=RuleDiscoveryMethod.DOMAIN_SPECIFIC,
                            confidence=confidence,
                            support=len(sample_values) / len(df),
                            evidence={
                                'domain': 'date',
                                'valid_count': valid_dates,
                                'total_count': len(sample_values)
                            },
                            rule_explanation="Date format validation based on column name pattern",
                            effectiveness_score=confidence
                        ))
                except Exception:
                    pass
        
        return discovered_rules[:self.config.max_rules_per_method]
    
    def _discover_distribution_rules(self, df: pd.DataFrame, dataset_id: str) -> List[DiscoveredRule]:
        """Discover rules based on data distribution analysis."""
        discovered_rules = []
        
        for column in df.select_dtypes(include=['int64', 'float64']).columns:
            values = df[column].dropna()
            
            if len(values) < 10:
                continue
            
            # Test for different distributions
            distributions = ['norm', 'uniform', 'exponential']
            
            for dist_name in distributions:
                try:
                    if dist_name == 'norm':
                        # Test for normal distribution
                        statistic, p_value = normaltest(values)
                        is_normal = p_value > 0.05
                        
                        if is_normal:
                            mean_val = values.mean()
                            std_val = values.std()
                            
                            rule = QualityRule(
                                rule_id=RuleId(),
                                rule_name=f"{column}_normal_distribution_validation",
                                description=f"Values in {column} should follow normal distribution",
                                category=RuleCategory.STATISTICAL_VALIDATION,
                                scope=RuleScope.COLUMN,
                                target_columns=[column],
                                validation_logic=ValidationLogic(
                                    logic_type=LogicType.STATISTICAL,
                                    expression=f"ABS(({column} - {mean_val}) / {std_val}) <= 3",
                                    parameters={
                                        'column_name': column,
                                        'distribution': 'normal',
                                        'mean': mean_val,
                                        'std': std_val
                                    },
                                    success_criteria=SuccessCriteria(min_pass_rate=0.95),
                                    error_message=f"Value violates normal distribution assumption"
                                ),
                                severity=Severity.LOW,
                                is_active=True
                            )
                            
                            confidence = min(1.0, 1 - p_value)
                            support = len(values) / len(df)
                            
                            discovered_rules.append(DiscoveredRule(
                                rule=rule,
                                discovery_method=RuleDiscoveryMethod.DISTRIBUTION_FITTING,
                                confidence=confidence,
                                support=support,
                                evidence={
                                    'distribution': dist_name,
                                    'test_statistic': statistic,
                                    'p_value': p_value,
                                    'parameters': {'mean': mean_val, 'std': std_val}
                                },
                                rule_explanation=f"Data follows {dist_name} distribution (p-value: {p_value:.4f})",
                                effectiveness_score=confidence * support * 0.5  # Lower weight for distribution rules
                            ))
                except Exception:
                    continue
        
        return discovered_rules[:self.config.max_rules_per_method]
    
    def _analyze_string_patterns(self, values: pd.Series) -> List[Dict[str, Any]]:
        """Analyze string patterns in values."""
        patterns = []
        
        # Common pattern templates
        pattern_templates = [
            {
                'name': 'email',
                'regex': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
                'description': 'Email address format'
            },
            {
                'name': 'phone',
                'regex': r'^[\d\s\-\(\)\+]{10,15}$',
                'description': 'Phone number format'
            },
            {
                'name': 'url',
                'regex': r'^https?://[^\s/$.?#].[^\s]*$',
                'description': 'URL format'
            },
            {
                'name': 'date',
                'regex': r'^\d{4}-\d{2}-\d{2}$',
                'description': 'Date format (YYYY-MM-DD)'
            },
            {
                'name': 'zipcode',
                'regex': r'^\d{5}(-\d{4})?$',
                'description': 'US ZIP code format'
            }
        ]
        
        for template in pattern_templates:
            matches = values.str.match(template['regex'], na=False).sum()
            if matches > 0:
                confidence = matches / len(values)
                patterns.append({
                    'name': template['name'],
                    'regex': template['regex'],
                    'description': template['description'],
                    'frequency': matches,
                    'confidence': confidence
                })
        
        # Analyze character patterns
        if len(values) > 0:
            # Length patterns
            lengths = values.str.len()
            most_common_length = lengths.mode().iloc[0] if len(lengths.mode()) > 0 else 0
            length_frequency = (lengths == most_common_length).sum()
            
            if length_frequency / len(values) > 0.8:
                patterns.append({
                    'name': 'fixed_length',
                    'regex': f'^.{{{most_common_length}}}$',
                    'description': f'Fixed length of {most_common_length} characters',
                    'frequency': length_frequency,
                    'confidence': length_frequency / len(values)
                })
        
        return sorted(patterns, key=lambda x: x['confidence'], reverse=True)
    
    def _remove_redundant_rules(self, rules: List[DiscoveredRule]) -> List[DiscoveredRule]:
        """Remove redundant rules."""
        unique_rules = []
        seen_signatures = set()
        
        for rule in rules:
            # Create signature for rule
            signature = (
                frozenset(rule.rule.target_columns),
                rule.rule.validation_logic.logic_type,
                rule.rule.category
            )
            
            if signature not in seen_signatures:
                unique_rules.append(rule)
                seen_signatures.add(signature)
        
        return unique_rules
    
    def _merge_similar_rules(self, rules: List[DiscoveredRule]) -> List[DiscoveredRule]:
        """Merge similar rules."""
        # Simple implementation - in practice, this would be more sophisticated
        merged_rules = []
        
        # Group rules by target columns and category
        rule_groups = defaultdict(list)
        for rule in rules:
            key = (frozenset(rule.rule.target_columns), rule.rule.category)
            rule_groups[key].append(rule)
        
        for group in rule_groups.values():
            if len(group) == 1:
                merged_rules.extend(group)
            else:
                # Merge rules with highest confidence
                best_rule = max(group, key=lambda x: x.confidence)
                merged_rules.append(best_rule)
        
        return merged_rules
    
    def _rank_rules(self, rules: List[DiscoveredRule]) -> List[DiscoveredRule]:
        """Rank rules by effectiveness."""
        # Calculate ranking score
        for rule in rules:
            rule.effectiveness_score = (
                rule.confidence * 0.4 +
                rule.support * 0.3 +
                (1.0 if rule.get_confidence_level() in [RuleConfidence.HIGH, RuleConfidence.VERY_HIGH] else 0.5) * 0.3
            )
        
        # Sort by effectiveness score
        return sorted(rules, key=lambda x: x.effectiveness_score, reverse=True)
    
    def _estimate_rule_coverage(self, rule: QualityRule, df: pd.DataFrame) -> float:
        """Estimate rule coverage."""
        if not rule.target_columns:
            return 1.0
        
        # Calculate coverage based on target columns
        applicable_rows = 0
        for column in rule.target_columns:
            if column in df.columns:
                applicable_rows += df[column].notna().sum()
        
        return applicable_rows / (len(df) * len(rule.target_columns)) if len(rule.target_columns) > 0 else 0.0
    
    def _calculate_rule_complexity(self, rule: QualityRule) -> float:
        """Calculate rule complexity score."""
        complexity = 0.0
        
        # Expression complexity
        expression_length = len(rule.validation_logic.expression)
        complexity += min(1.0, expression_length / 100)
        
        # Number of target columns
        complexity += len(rule.target_columns) * 0.1
        
        # Logic type complexity
        logic_complexity = {
            LogicType.REGEX: 0.8,
            LogicType.PYTHON: 0.9,
            LogicType.SQL: 0.7,
            LogicType.STATISTICAL: 0.6,
            LogicType.COMPARISON: 0.3,
            LogicType.AGGREGATION: 0.5,
            LogicType.LOOKUP: 0.2,
            LogicType.CONDITIONAL: 0.8,
            LogicType.EXPRESSION: 0.7
        }
        complexity += logic_complexity.get(rule.validation_logic.logic_type, 0.5)
        
        return min(1.0, complexity)
    
    def _calculate_maintainability(self, rule: QualityRule) -> float:
        """Calculate rule maintainability score."""
        maintainability = 1.0
        
        # Reduce score for complex expressions
        if len(rule.validation_logic.expression) > 200:
            maintainability -= 0.3
        
        # Reduce score for multiple target columns
        if len(rule.target_columns) > 3:
            maintainability -= 0.2
        
        # Reduce score for complex logic types
        if rule.validation_logic.logic_type in [LogicType.PYTHON, LogicType.CONDITIONAL]:
            maintainability -= 0.2
        
        return max(0.0, maintainability)
    
    def _estimate_business_value(self, rule: QualityRule) -> float:
        """Estimate business value of rule."""
        value = 0.5  # Base value
        
        # Higher value for critical categories
        high_value_categories = [
            RuleCategory.BUSINESS_RULE_VALIDATION,
            RuleCategory.UNIQUENESS_VALIDATION,
            RuleCategory.COMPLETENESS_VALIDATION
        ]
        
        if rule.category in high_value_categories:
            value += 0.3
        
        # Higher value for high severity
        if rule.severity == Severity.CRITICAL:
            value += 0.3
        elif rule.severity == Severity.HIGH:
            value += 0.2
        
        return min(1.0, value)
    
    def _initialize_rule_templates(self) -> Dict[str, Any]:
        """Initialize rule templates."""
        return {
            'completeness': {
                'expression': "df['{column}'].notna()",
                'description': "Check for completeness",
                'severity': Severity.MEDIUM
            },
            'uniqueness': {
                'expression': "~df['{column}'].duplicated()",
                'description': "Check for uniqueness",
                'severity': Severity.HIGH
            },
            'range': {
                'expression': "df['{column}'].between({min_val}, {max_val})",
                'description': "Check value range",
                'severity': Severity.MEDIUM
            }
        }
    
    def _initialize_domain_patterns(self) -> Dict[str, str]:
        """Initialize domain-specific patterns."""
        return {
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'phone': r'^[\d\s\-\(\)\+]{10,15}$',
            'url': r'^https?://[^\s/$.?#].[^\s]*$',
            'ipv4': r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$',
            'ssn': r'^\d{3}-\d{2}-\d{4}$',
            'credit_card': r'^\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}$',
            'zipcode': r'^\d{5}(-\d{4})?$'
        }
    
    def _update_discovery_stats(self, discovered_rules: List[DiscoveredRule]) -> None:
        """Update discovery statistics."""
        self._discovery_stats['total_discoveries'] += 1
        self._discovery_stats['rules_discovered'] += len(discovered_rules)
        
        high_quality_rules = [rule for rule in discovered_rules if rule.is_high_quality()]
        self._discovery_stats['high_quality_rules'] += len(high_quality_rules)
        
        if discovered_rules:
            confidences = [rule.confidence for rule in discovered_rules]
            supports = [rule.support for rule in discovered_rules]
            
            self._discovery_stats['avg_confidence'] = mean(confidences)
            self._discovery_stats['avg_support'] = mean(supports)
    
    def get_discovery_statistics(self) -> Dict[str, Any]:
        """Get discovery service statistics."""
        return self._discovery_stats.copy()