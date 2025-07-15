"""
Cleansing Impact Analysis Framework
Measures data quality improvement, business impact, ROI analysis, and effectiveness monitoring.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import uuid
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Comprehensive quality metrics for before/after comparison."""
    
    # Completeness metrics
    completeness_rate: float
    missing_values_count: int
    missing_percentage: float
    
    # Accuracy metrics
    accuracy_score: float
    error_rate: float
    outlier_count: int
    outlier_percentage: float
    
    # Consistency metrics
    consistency_score: float
    format_inconsistencies: int
    duplicate_count: int
    duplicate_percentage: float
    
    # Validity metrics
    validity_score: float
    invalid_values_count: int
    constraint_violations: int
    
    # Timeliness metrics
    timeliness_score: float
    outdated_records: int
    
    # Overall metrics
    overall_quality_score: float
    data_size_mb: float
    record_count: int
    column_count: int


@dataclass
class BusinessImpactMetrics:
    """Business impact metrics from data cleansing."""
    
    # Operational impact
    processing_time_improvement_seconds: float
    error_reduction_percentage: float
    manual_intervention_reduction: int
    
    # Decision making impact
    decision_accuracy_improvement: float
    insights_reliability_score: float
    report_accuracy_improvement: float
    
    # Cost impact
    cost_of_poor_data_before: float
    cost_of_poor_data_after: float
    cost_savings: float
    cleansing_cost: float
    roi_percentage: float
    
    # Compliance impact
    compliance_score_improvement: float
    risk_reduction_score: float
    
    # Customer impact
    customer_satisfaction_improvement: float
    data_driven_decisions_count: int


@dataclass
class StatisticalSignificance:
    """Statistical significance analysis of quality improvements."""
    
    test_name: str
    statistic: float
    p_value: float
    confidence_level: float
    is_significant: bool
    effect_size: float
    interpretation: str


@dataclass
class CleansingImpactReport:
    """Comprehensive cleansing impact analysis report."""
    
    analysis_id: str
    analysis_date: datetime
    cleansing_operation_id: str
    
    # Quality analysis
    before_quality_metrics: QualityMetrics
    after_quality_metrics: QualityMetrics
    quality_improvement_delta: Dict[str, float]
    
    # Statistical analysis
    statistical_tests: List[StatisticalSignificance]
    confidence_intervals: Dict[str, Tuple[float, float]]
    
    # Business impact
    business_impact: BusinessImpactMetrics
    
    # Performance analysis
    processing_performance: Dict[str, float]
    scalability_metrics: Dict[str, float]
    
    # Recommendations
    recommendations: List[str]
    risk_assessments: List[str]
    
    # Audit trail
    analysis_metadata: Dict[str, Any]
    data_lineage: List[str]


class DataQualityAnalyzer:
    """Analyzes data quality metrics for before/after comparison."""
    
    def __init__(self):
        self.quality_dimensions = [
            'completeness', 'accuracy', 'consistency', 
            'validity', 'timeliness', 'uniqueness'
        ]
    
    def analyze_quality_metrics(self, df: pd.DataFrame, 
                              reference_data: Optional[pd.DataFrame] = None) -> QualityMetrics:
        """Calculate comprehensive quality metrics for a dataset."""
        
        total_cells = len(df) * len(df.columns)
        missing_count = df.isnull().sum().sum()
        
        # Completeness metrics
        completeness_rate = 1 - (missing_count / total_cells)
        missing_percentage = (missing_count / total_cells) * 100
        
        # Accuracy metrics (outlier detection)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        outlier_count = 0
        
        for col in numeric_columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_mask = (df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))
            outlier_count += outlier_mask.sum()
        
        outlier_percentage = (outlier_count / len(df)) * 100 if len(df) > 0 else 0
        accuracy_score = 1 - (outlier_percentage / 100)
        error_rate = outlier_percentage / 100
        
        # Consistency metrics
        duplicate_count = df.duplicated().sum()
        duplicate_percentage = (duplicate_count / len(df)) * 100 if len(df) > 0 else 0
        
        # Format consistency for object columns
        format_inconsistencies = 0
        for col in df.select_dtypes(include=['object']).columns:
            if len(df[col].dropna()) > 0:
                # Check format patterns
                patterns = df[col].dropna().astype(str).apply(self._extract_format_pattern)
                pattern_counts = patterns.value_counts()
                if len(pattern_counts) > 1:
                    # Multiple formats found
                    format_inconsistencies += len(pattern_counts) - 1
        
        consistency_score = 1 - (duplicate_percentage / 100) - (format_inconsistencies / len(df.columns))
        consistency_score = max(0, min(1, consistency_score))
        
        # Validity metrics (basic validation)
        invalid_values_count = 0
        constraint_violations = 0
        
        # Check for basic validity issues
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check for obviously invalid values
                invalid_patterns = df[col].astype(str).str.contains(r'[<>{}[\]\\|;]', regex=True, na=False)
                invalid_values_count += invalid_patterns.sum()
        
        validity_score = 1 - (invalid_values_count / len(df)) if len(df) > 0 else 1
        
        # Timeliness (placeholder - would need timestamp columns)
        timeliness_score = 1.0  # Assume current unless proven otherwise
        outdated_records = 0
        
        # Overall quality score (weighted average)
        overall_quality_score = (
            completeness_rate * 0.25 +
            accuracy_score * 0.25 +
            consistency_score * 0.25 +
            validity_score * 0.15 +
            timeliness_score * 0.10
        )
        
        # Data size metrics
        data_size_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        
        return QualityMetrics(
            completeness_rate=completeness_rate,
            missing_values_count=missing_count,
            missing_percentage=missing_percentage,
            accuracy_score=accuracy_score,
            error_rate=error_rate,
            outlier_count=outlier_count,
            outlier_percentage=outlier_percentage,
            consistency_score=consistency_score,
            format_inconsistencies=format_inconsistencies,
            duplicate_count=duplicate_count,
            duplicate_percentage=duplicate_percentage,
            validity_score=validity_score,
            invalid_values_count=invalid_values_count,
            constraint_violations=constraint_violations,
            timeliness_score=timeliness_score,
            outdated_records=outdated_records,
            overall_quality_score=overall_quality_score,
            data_size_mb=data_size_mb,
            record_count=len(df),
            column_count=len(df.columns)
        )
    
    def _extract_format_pattern(self, value: str) -> str:
        """Extract format pattern from string value."""
        import re
        pattern = re.sub(r'\d', 'N', str(value))  # Replace digits with N
        pattern = re.sub(r'[a-zA-Z]', 'A', pattern)  # Replace letters with A
        return pattern


class StatisticalAnalyzer:
    """Performs statistical analysis of cleansing impact."""
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
    
    def test_quality_improvement_significance(self, 
                                           before_metrics: QualityMetrics,
                                           after_metrics: QualityMetrics,
                                           sample_size: int = None) -> List[StatisticalSignificance]:
        """Test statistical significance of quality improvements."""
        
        tests = []
        
        # Test overall quality improvement
        quality_improvement = after_metrics.overall_quality_score - before_metrics.overall_quality_score
        
        # Simulated statistical test (in real implementation, would use actual sample data)
        # For demonstration, using a simple z-test approximation
        
        if sample_size and sample_size > 30:
            # Large sample z-test
            standard_error = np.sqrt((before_metrics.overall_quality_score * (1 - before_metrics.overall_quality_score)) / sample_size)
            z_statistic = quality_improvement / standard_error if standard_error > 0 else 0
            p_value = 2 * (1 - stats.norm.cdf(abs(z_statistic)))
            
            tests.append(StatisticalSignificance(
                test_name="Overall Quality Improvement Z-Test",
                statistic=z_statistic,
                p_value=p_value,
                confidence_level=self.confidence_level,
                is_significant=p_value < self.alpha,
                effect_size=quality_improvement,
                interpretation=self._interpret_significance(p_value, quality_improvement)
            ))
        
        # Test completeness improvement
        completeness_improvement = after_metrics.completeness_rate - before_metrics.completeness_rate
        
        # McNemar's test for paired proportions (approximated)
        if sample_size:
            chi_square = (completeness_improvement ** 2) * sample_size
            p_value_completeness = 1 - stats.chi2.cdf(chi_square, df=1)
            
            tests.append(StatisticalSignificance(
                test_name="Completeness Improvement Test",
                statistic=chi_square,
                p_value=p_value_completeness,
                confidence_level=self.confidence_level,
                is_significant=p_value_completeness < self.alpha,
                effect_size=completeness_improvement,
                interpretation=self._interpret_significance(p_value_completeness, completeness_improvement)
            ))
        
        # Test duplicate reduction
        duplicate_reduction = before_metrics.duplicate_percentage - after_metrics.duplicate_percentage
        
        if sample_size:
            z_stat_dup = duplicate_reduction / np.sqrt(before_metrics.duplicate_percentage / 100 / sample_size) if before_metrics.duplicate_percentage > 0 else 0
            p_value_dup = 2 * (1 - stats.norm.cdf(abs(z_stat_dup)))
            
            tests.append(StatisticalSignificance(
                test_name="Duplicate Reduction Test",
                statistic=z_stat_dup,
                p_value=p_value_dup,
                confidence_level=self.confidence_level,
                is_significant=p_value_dup < self.alpha,
                effect_size=duplicate_reduction,
                interpretation=self._interpret_significance(p_value_dup, duplicate_reduction)
            ))
        
        return tests
    
    def calculate_confidence_intervals(self, 
                                     before_metrics: QualityMetrics,
                                     after_metrics: QualityMetrics,
                                     sample_size: int = None) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for quality improvements."""
        
        intervals = {}
        
        if sample_size and sample_size > 30:
            z_critical = stats.norm.ppf(1 - self.alpha/2)
            
            # Overall quality improvement CI
            quality_diff = after_metrics.overall_quality_score - before_metrics.overall_quality_score
            se_quality = np.sqrt(
                (before_metrics.overall_quality_score * (1 - before_metrics.overall_quality_score) +
                 after_metrics.overall_quality_score * (1 - after_metrics.overall_quality_score)) / sample_size
            )
            
            intervals['overall_quality_improvement'] = (
                quality_diff - z_critical * se_quality,
                quality_diff + z_critical * se_quality
            )
            
            # Completeness improvement CI
            completeness_diff = after_metrics.completeness_rate - before_metrics.completeness_rate
            se_completeness = np.sqrt(
                (before_metrics.completeness_rate * (1 - before_metrics.completeness_rate) +
                 after_metrics.completeness_rate * (1 - after_metrics.completeness_rate)) / sample_size
            )
            
            intervals['completeness_improvement'] = (
                completeness_diff - z_critical * se_completeness,
                completeness_diff + z_critical * se_completeness
            )
        
        return intervals
    
    def _interpret_significance(self, p_value: float, effect_size: float) -> str:
        """Interpret statistical significance results."""
        
        if p_value < 0.001:
            significance = "highly significant"
        elif p_value < 0.01:
            significance = "very significant"
        elif p_value < 0.05:
            significance = "significant"
        else:
            significance = "not significant"
        
        if abs(effect_size) < 0.01:
            magnitude = "negligible"
        elif abs(effect_size) < 0.05:
            magnitude = "small"
        elif abs(effect_size) < 0.10:
            magnitude = "medium"
        else:
            magnitude = "large"
        
        direction = "improvement" if effect_size > 0 else "deterioration"
        
        return f"The quality {direction} is {significance} (p={p_value:.4f}) with {magnitude} effect size ({effect_size:.4f})"


class BusinessImpactCalculator:
    """Calculates business impact and ROI from data cleansing."""
    
    def __init__(self):
        # Default cost factors (can be customized per organization)
        self.default_cost_factors = {
            'cost_per_error': 10.0,  # Cost per data error
            'cost_per_missing_value': 5.0,  # Cost per missing value
            'cost_per_duplicate': 2.0,  # Cost per duplicate record
            'hourly_analyst_cost': 50.0,  # Cost per analyst hour
            'decision_delay_cost': 100.0,  # Cost per delayed decision
            'compliance_violation_cost': 1000.0  # Cost per compliance issue
        }
    
    def calculate_business_impact(self, 
                                before_metrics: QualityMetrics,
                                after_metrics: QualityMetrics,
                                cleansing_cost: float,
                                custom_cost_factors: Dict[str, float] = None) -> BusinessImpactMetrics:
        """Calculate comprehensive business impact from cleansing."""
        
        cost_factors = {**self.default_cost_factors, **(custom_cost_factors or {})}
        
        # Calculate cost of poor data before and after
        cost_before = self._calculate_poor_data_cost(before_metrics, cost_factors)
        cost_after = self._calculate_poor_data_cost(after_metrics, cost_factors)
        
        cost_savings = cost_before - cost_after
        roi_percentage = ((cost_savings - cleansing_cost) / cleansing_cost * 100) if cleansing_cost > 0 else 0
        
        # Operational impact
        processing_time_improvement = self._estimate_processing_time_improvement(before_metrics, after_metrics)
        error_reduction = ((before_metrics.error_rate - after_metrics.error_rate) / before_metrics.error_rate * 100) if before_metrics.error_rate > 0 else 0
        manual_intervention_reduction = int((before_metrics.missing_values_count - after_metrics.missing_values_count) * 0.1)  # Estimate
        
        # Decision making impact
        decision_accuracy_improvement = (after_metrics.overall_quality_score - before_metrics.overall_quality_score) * 0.5  # Estimate
        insights_reliability_score = after_metrics.overall_quality_score * 0.9  # Estimate
        report_accuracy_improvement = (after_metrics.accuracy_score - before_metrics.accuracy_score) * 0.8  # Estimate
        
        # Compliance impact
        compliance_improvement = self._calculate_compliance_improvement(before_metrics, after_metrics)
        risk_reduction = compliance_improvement * 0.7  # Estimate
        
        # Customer impact (estimated)
        customer_satisfaction_improvement = decision_accuracy_improvement * 0.6  # Estimate
        data_driven_decisions_count = int(after_metrics.overall_quality_score * 100)  # Estimate
        
        return BusinessImpactMetrics(
            processing_time_improvement_seconds=processing_time_improvement,
            error_reduction_percentage=error_reduction,
            manual_intervention_reduction=manual_intervention_reduction,
            decision_accuracy_improvement=decision_accuracy_improvement,
            insights_reliability_score=insights_reliability_score,
            report_accuracy_improvement=report_accuracy_improvement,
            cost_of_poor_data_before=cost_before,
            cost_of_poor_data_after=cost_after,
            cost_savings=cost_savings,
            cleansing_cost=cleansing_cost,
            roi_percentage=roi_percentage,
            compliance_score_improvement=compliance_improvement,
            risk_reduction_score=risk_reduction,
            customer_satisfaction_improvement=customer_satisfaction_improvement,
            data_driven_decisions_count=data_driven_decisions_count
        )
    
    def _calculate_poor_data_cost(self, metrics: QualityMetrics, cost_factors: Dict[str, float]) -> float:
        """Calculate the cost of poor data quality."""
        
        cost = 0.0
        
        # Cost of missing values
        cost += metrics.missing_values_count * cost_factors['cost_per_missing_value']
        
        # Cost of duplicates
        cost += metrics.duplicate_count * cost_factors['cost_per_duplicate']
        
        # Cost of errors (outliers and invalid values)
        total_errors = metrics.outlier_count + metrics.invalid_values_count
        cost += total_errors * cost_factors['cost_per_error']
        
        # Cost of format inconsistencies (analyst time)
        cost += metrics.format_inconsistencies * cost_factors['hourly_analyst_cost'] * 0.5  # 30 min per inconsistency
        
        # Cost of constraint violations
        cost += metrics.constraint_violations * cost_factors['compliance_violation_cost']
        
        return cost
    
    def _estimate_processing_time_improvement(self, before: QualityMetrics, after: QualityMetrics) -> float:
        """Estimate processing time improvement from quality enhancement."""
        
        # Simple heuristic: better quality = faster processing
        quality_improvement = after.overall_quality_score - before.overall_quality_score
        
        # Assume 1% quality improvement = 2% processing time improvement
        time_improvement_percentage = quality_improvement * 2
        
        # Estimate based on data size (larger datasets benefit more)
        base_processing_time = before.data_size_mb * 0.1  # 0.1 seconds per MB
        improvement_seconds = base_processing_time * time_improvement_percentage
        
        return improvement_seconds
    
    def _calculate_compliance_improvement(self, before: QualityMetrics, after: QualityMetrics) -> float:
        """Calculate compliance score improvement."""
        
        # Focus on completeness, validity, and consistency for compliance
        before_compliance = (before.completeness_rate + before.validity_score + before.consistency_score) / 3
        after_compliance = (after.completeness_rate + after.validity_score + after.consistency_score) / 3
        
        return after_compliance - before_compliance


class CleansingImpactAnalysisEngine:
    """
    Comprehensive engine for analyzing the impact of data cleansing operations.
    Provides quality improvement analysis, statistical significance testing, and business impact calculation.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        self.quality_analyzer = DataQualityAnalyzer()
        self.statistical_analyzer = StatisticalAnalyzer(confidence_level)
        self.business_calculator = BusinessImpactCalculator()
        
        logger.info("Initialized CleansingImpactAnalysisEngine")
    
    def analyze_cleansing_impact(self,
                               before_data: pd.DataFrame,
                               after_data: pd.DataFrame,
                               cleansing_cost: float,
                               operation_id: str = None,
                               custom_cost_factors: Dict[str, float] = None) -> CleansingImpactReport:
        """
        Perform comprehensive analysis of cleansing impact.
        
        Args:
            before_data: Dataset before cleansing
            after_data: Dataset after cleansing
            cleansing_cost: Cost of the cleansing operation
            operation_id: ID of the cleansing operation
            custom_cost_factors: Custom business cost factors
        
        Returns:
            Comprehensive impact analysis report
        """
        
        analysis_id = str(uuid.uuid4())
        operation_id = operation_id or f"cleansing_{int(datetime.now().timestamp())}"
        
        logger.info(f"Starting cleansing impact analysis {analysis_id}")
        
        try:
            # Calculate quality metrics
            before_quality = self.quality_analyzer.analyze_quality_metrics(before_data)
            after_quality = self.quality_analyzer.analyze_quality_metrics(after_data)
            
            # Calculate quality improvement deltas
            quality_deltas = self._calculate_quality_deltas(before_quality, after_quality)
            
            # Statistical significance testing
            statistical_tests = self.statistical_analyzer.test_quality_improvement_significance(
                before_quality, after_quality, len(before_data)
            )
            
            # Confidence intervals
            confidence_intervals = self.statistical_analyzer.calculate_confidence_intervals(
                before_quality, after_quality, len(before_data)
            )
            
            # Business impact analysis
            business_impact = self.business_calculator.calculate_business_impact(
                before_quality, after_quality, cleansing_cost, custom_cost_factors
            )
            
            # Performance analysis
            performance_metrics = self._analyze_performance(before_data, after_data)
            
            # Scalability metrics
            scalability_metrics = self._analyze_scalability(before_data, after_data)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                before_quality, after_quality, business_impact, statistical_tests
            )
            
            # Risk assessments
            risk_assessments = self._assess_risks(after_quality, business_impact)
            
            # Analysis metadata
            metadata = {
                'analysis_timestamp': datetime.now(),
                'before_data_shape': before_data.shape,
                'after_data_shape': after_data.shape,
                'analysis_duration_seconds': 0,  # Would be calculated in real implementation
                'confidence_level': self.statistical_analyzer.confidence_level,
                'cost_factors_used': custom_cost_factors or self.business_calculator.default_cost_factors
            }
            
            # Data lineage
            lineage = [
                f"Original data: {before_data.shape[0]} rows, {before_data.shape[1]} columns",
                f"Cleansing operation: {operation_id}",
                f"Cleansed data: {after_data.shape[0]} rows, {after_data.shape[1]} columns"
            ]
            
            report = CleansingImpactReport(
                analysis_id=analysis_id,
                analysis_date=datetime.now(),
                cleansing_operation_id=operation_id,
                before_quality_metrics=before_quality,
                after_quality_metrics=after_quality,
                quality_improvement_delta=quality_deltas,
                statistical_tests=statistical_tests,
                confidence_intervals=confidence_intervals,
                business_impact=business_impact,
                processing_performance=performance_metrics,
                scalability_metrics=scalability_metrics,
                recommendations=recommendations,
                risk_assessments=risk_assessments,
                analysis_metadata=metadata,
                data_lineage=lineage
            )
            
            logger.info(f"Cleansing impact analysis completed. Overall quality improvement: {quality_deltas.get('overall_quality_score', 0):.2%}")
            
            return report
            
        except Exception as e:
            logger.error(f"Cleansing impact analysis failed: {e}")
            raise
    
    def _calculate_quality_deltas(self, before: QualityMetrics, after: QualityMetrics) -> Dict[str, float]:
        """Calculate quality improvement deltas."""
        
        deltas = {
            'overall_quality_score': after.overall_quality_score - before.overall_quality_score,
            'completeness_rate': after.completeness_rate - before.completeness_rate,
            'accuracy_score': after.accuracy_score - before.accuracy_score,
            'consistency_score': after.consistency_score - before.consistency_score,
            'validity_score': after.validity_score - before.validity_score,
            'timeliness_score': after.timeliness_score - before.timeliness_score,
            'missing_values_reduction': before.missing_values_count - after.missing_values_count,
            'duplicate_reduction': before.duplicate_count - after.duplicate_count,
            'outlier_reduction': before.outlier_count - after.outlier_count,
            'error_rate_reduction': before.error_rate - after.error_rate
        }
        
        return deltas
    
    def _analyze_performance(self, before_data: pd.DataFrame, after_data: pd.DataFrame) -> Dict[str, float]:
        """Analyze processing performance improvements."""
        
        # Simple performance metrics (would be more sophisticated in real implementation)
        return {
            'data_size_change_mb': (after_data.memory_usage(deep=True).sum() - before_data.memory_usage(deep=True).sum()) / 1024 / 1024,
            'row_count_change': len(after_data) - len(before_data),
            'column_count_change': len(after_data.columns) - len(before_data.columns),
            'compression_ratio': len(after_data) / len(before_data) if len(before_data) > 0 else 1,
            'estimated_query_speedup': 1.1  # Placeholder estimate
        }
    
    def _analyze_scalability(self, before_data: pd.DataFrame, after_data: pd.DataFrame) -> Dict[str, float]:
        """Analyze scalability implications."""
        
        return {
            'memory_efficiency_improvement': 0.05,  # Placeholder
            'processing_scalability_factor': 1.2,  # Placeholder
            'storage_efficiency_improvement': 0.1,  # Placeholder
            'network_transfer_improvement': 0.15   # Placeholder
        }
    
    def _generate_recommendations(self, 
                                before_quality: QualityMetrics,
                                after_quality: QualityMetrics,
                                business_impact: BusinessImpactMetrics,
                                statistical_tests: List[StatisticalSignificance]) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        
        recommendations = []
        
        # Quality-based recommendations
        if after_quality.overall_quality_score > 0.9:
            recommendations.append("Excellent data quality achieved - consider implementing monitoring to maintain standards")
        elif after_quality.overall_quality_score > 0.8:
            recommendations.append("Good data quality achieved - focus on remaining quality issues for further improvement")
        else:
            recommendations.append("Data quality needs improvement - consider additional cleansing strategies")
        
        # Business impact recommendations
        if business_impact.roi_percentage > 200:
            recommendations.append("High ROI achieved - expand cleansing program to other datasets")
        elif business_impact.roi_percentage > 100:
            recommendations.append("Positive ROI achieved - continue current cleansing approach")
        else:
            recommendations.append("ROI below expectations - review cost-effectiveness of cleansing approach")
        
        # Statistical significance recommendations
        significant_tests = [test for test in statistical_tests if test.is_significant]
        if len(significant_tests) == len(statistical_tests):
            recommendations.append("All quality improvements are statistically significant - results are reliable")
        elif len(significant_tests) > 0:
            recommendations.append("Some quality improvements are statistically significant - focus on validated improvements")
        else:
            recommendations.append("Quality improvements lack statistical significance - consider larger sample sizes or different approaches")
        
        # Specific quality dimension recommendations
        if after_quality.completeness_rate < 0.9:
            recommendations.append("Address remaining missing values through imputation or collection strategies")
        
        if after_quality.duplicate_percentage > 5:
            recommendations.append("Implement stronger duplicate detection and prevention measures")
        
        if after_quality.consistency_score < 0.8:
            recommendations.append("Standardize data formats and validation rules to improve consistency")
        
        return recommendations
    
    def _assess_risks(self, after_quality: QualityMetrics, 
                     business_impact: BusinessImpactMetrics) -> List[str]:
        """Assess risks and potential issues."""
        
        risks = []
        
        # Quality risks
        if after_quality.overall_quality_score < 0.7:
            risks.append("Low overall quality score may impact decision-making reliability")
        
        if after_quality.completeness_rate < 0.8:
            risks.append("High missing value rate may bias analysis results")
        
        if after_quality.accuracy_score < 0.8:
            risks.append("Low accuracy score may lead to incorrect insights")
        
        # Business risks
        if business_impact.roi_percentage < 50:
            risks.append("Low ROI may not justify continued investment in cleansing")
        
        if business_impact.compliance_score_improvement < 0.1:
            risks.append("Limited compliance improvement may leave regulatory risks")
        
        # Operational risks
        if business_impact.processing_time_improvement_seconds < 0:
            risks.append("Processing time may have increased - review cleansing efficiency")
        
        return risks
    
    def generate_impact_visualization(self, report: CleansingImpactReport, 
                                    output_path: Optional[str] = None) -> Dict[str, Any]:
        """Generate visualizations for impact analysis."""
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'Data Cleansing Impact Analysis - {report.analysis_id}', fontsize=16, fontweight='bold')
            
            # 1. Quality metrics comparison
            quality_metrics = ['Overall Quality', 'Completeness', 'Accuracy', 'Consistency', 'Validity']
            before_scores = [
                report.before_quality_metrics.overall_quality_score,
                report.before_quality_metrics.completeness_rate,
                report.before_quality_metrics.accuracy_score,
                report.before_quality_metrics.consistency_score,
                report.before_quality_metrics.validity_score
            ]
            after_scores = [
                report.after_quality_metrics.overall_quality_score,
                report.after_quality_metrics.completeness_rate,
                report.after_quality_metrics.accuracy_score,
                report.after_quality_metrics.consistency_score,
                report.after_quality_metrics.validity_score
            ]
            
            x = np.arange(len(quality_metrics))
            width = 0.35
            
            axes[0, 0].bar(x - width/2, before_scores, width, label='Before', alpha=0.8)
            axes[0, 0].bar(x + width/2, after_scores, width, label='After', alpha=0.8)
            axes[0, 0].set_title('Quality Metrics Comparison')
            axes[0, 0].set_ylabel('Score')
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels(quality_metrics, rotation=45)
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Business impact metrics
            impact_metrics = ['Cost Savings', 'ROI %', 'Error Reduction %']
            impact_values = [
                report.business_impact.cost_savings,
                report.business_impact.roi_percentage,
                report.business_impact.error_reduction_percentage
            ]
            
            axes[0, 1].bar(impact_metrics, impact_values, color=['green', 'blue', 'orange'], alpha=0.7)
            axes[0, 1].set_title('Business Impact Metrics')
            axes[0, 1].set_ylabel('Value')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Data issues reduction
            issue_types = ['Missing Values', 'Duplicates', 'Outliers', 'Invalid Values']
            before_issues = [
                report.before_quality_metrics.missing_values_count,
                report.before_quality_metrics.duplicate_count,
                report.before_quality_metrics.outlier_count,
                report.before_quality_metrics.invalid_values_count
            ]
            after_issues = [
                report.after_quality_metrics.missing_values_count,
                report.after_quality_metrics.duplicate_count,
                report.after_quality_metrics.outlier_count,
                report.after_quality_metrics.invalid_values_count
            ]
            
            x = np.arange(len(issue_types))
            axes[0, 2].bar(x - width/2, before_issues, width, label='Before', alpha=0.8, color='red')
            axes[0, 2].bar(x + width/2, after_issues, width, label='After', alpha=0.8, color='green')
            axes[0, 2].set_title('Data Issues Reduction')
            axes[0, 2].set_ylabel('Count')
            axes[0, 2].set_xticks(x)
            axes[0, 2].set_xticklabels(issue_types, rotation=45)
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
            
            # 4. Statistical significance
            test_names = [test.test_name.split()[0] for test in report.statistical_tests]
            p_values = [test.p_value for test in report.statistical_tests]
            significance_line = [0.05] * len(test_names)
            
            axes[1, 0].bar(test_names, p_values, alpha=0.7, color='purple')
            axes[1, 0].plot(test_names, significance_line, 'r--', label='α=0.05')
            axes[1, 0].set_title('Statistical Significance (p-values)')
            axes[1, 0].set_ylabel('p-value')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # 5. Quality improvement over time (simulated)
            time_points = ['Week 1', 'Week 2', 'Week 3', 'Week 4']
            quality_progression = [
                report.before_quality_metrics.overall_quality_score,
                report.before_quality_metrics.overall_quality_score + 0.3 * report.quality_improvement_delta['overall_quality_score'],
                report.before_quality_metrics.overall_quality_score + 0.7 * report.quality_improvement_delta['overall_quality_score'],
                report.after_quality_metrics.overall_quality_score
            ]
            
            axes[1, 1].plot(time_points, quality_progression, marker='o', linewidth=2, markersize=8)
            axes[1, 1].set_title('Quality Improvement Timeline')
            axes[1, 1].set_ylabel('Overall Quality Score')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_ylim(0, 1)
            
            # 6. ROI Analysis
            costs = ['Cleansing Cost', 'Cost of Poor Data (Before)', 'Cost of Poor Data (After)']
            cost_values = [
                report.business_impact.cleansing_cost,
                report.business_impact.cost_of_poor_data_before,
                report.business_impact.cost_of_poor_data_after
            ]
            
            colors = ['red', 'orange', 'green']
            axes[1, 2].bar(costs, cost_values, color=colors, alpha=0.7)
            axes[1, 2].set_title('Cost Analysis')
            axes[1, 2].set_ylabel('Cost ($)')
            axes[1, 2].tick_params(axis='x', rotation=45)
            axes[1, 2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Impact visualization saved to {output_path}")
            
            plt.show()
            
            return {
                'visualization_created': True,
                'output_path': output_path,
                'charts_generated': 6
            }
            
        except ImportError:
            logger.warning("Matplotlib/Seaborn not available - skipping visualization")
            return {'visualization_created': False, 'error': 'Visualization libraries not available'}
        except Exception as e:
            logger.error(f"Visualization generation failed: {e}")
            return {'visualization_created': False, 'error': str(e)}
    
    def export_impact_report(self, report: CleansingImpactReport, 
                           output_format: str = 'json',
                           output_path: str = None) -> str:
        """Export impact report to various formats."""
        
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"cleansing_impact_report_{timestamp}.{output_format}"
        
        try:
            if output_format.lower() == 'json':
                import json
                
                # Convert dataclass to dict (simplified)
                report_dict = {
                    'analysis_id': report.analysis_id,
                    'analysis_date': report.analysis_date.isoformat(),
                    'cleansing_operation_id': report.cleansing_operation_id,
                    'quality_improvement_summary': {
                        'overall_quality_improvement': report.quality_improvement_delta.get('overall_quality_score', 0),
                        'completeness_improvement': report.quality_improvement_delta.get('completeness_rate', 0),
                        'accuracy_improvement': report.quality_improvement_delta.get('accuracy_score', 0)
                    },
                    'business_impact_summary': {
                        'roi_percentage': report.business_impact.roi_percentage,
                        'cost_savings': report.business_impact.cost_savings,
                        'error_reduction_percentage': report.business_impact.error_reduction_percentage
                    },
                    'recommendations': report.recommendations,
                    'risk_assessments': report.risk_assessments
                }
                
                with open(output_path, 'w') as f:
                    json.dump(report_dict, f, indent=2)
            
            elif output_format.lower() == 'csv':
                # Create summary CSV
                summary_data = {
                    'Metric': [
                        'Overall Quality Improvement',
                        'Completeness Improvement',
                        'Accuracy Improvement',
                        'ROI Percentage',
                        'Cost Savings',
                        'Error Reduction Percentage'
                    ],
                    'Value': [
                        report.quality_improvement_delta.get('overall_quality_score', 0),
                        report.quality_improvement_delta.get('completeness_rate', 0),
                        report.quality_improvement_delta.get('accuracy_score', 0),
                        report.business_impact.roi_percentage,
                        report.business_impact.cost_savings,
                        report.business_impact.error_reduction_percentage
                    ]
                }
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_csv(output_path, index=False)
            
            else:
                raise ValueError(f"Unsupported output format: {output_format}")
            
            logger.info(f"Impact report exported to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Report export failed: {e}")
            raise