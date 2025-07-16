"""Quality Optimization Engine Service.

Service for providing automated quality improvement recommendations,
optimization strategies, and actionable insights for data quality enhancement.
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
from abc import ABC, abstractmethod

from ...domain.entities.quality_profile import DataQualityProfile, DatasetId
from ...domain.entities.quality_scores import QualityScores
from ...domain.entities.quality_anomaly import QualityAnomaly, AnomalySeverity
from ...domain.entities.validation_rule import QualityRule, RuleCategory, Severity
from .quality_assessment_service import QualityAssessmentService
from .advanced_quality_metrics_service import AdvancedQualityScore, MetricType

logger = logging.getLogger(__name__)


class OptimizationType(Enum):
    """Types of quality optimization."""
    IMMEDIATE_FIX = "immediate_fix"
    PROCESS_IMPROVEMENT = "process_improvement"
    INFRASTRUCTURE_UPGRADE = "infrastructure_upgrade"
    MONITORING_ENHANCEMENT = "monitoring_enhancement"
    PREVENTION_STRATEGY = "prevention_strategy"
    AUTOMATION_OPPORTUNITY = "automation_opportunity"


class OptimizationPriority(Enum):
    """Priority levels for optimization recommendations."""
    CRITICAL = "critical"      # Immediate action required
    HIGH = "high"             # Address within 1 week
    MEDIUM = "medium"         # Address within 1 month
    LOW = "low"              # Address when convenient
    NICE_TO_HAVE = "nice_to_have"  # Optional enhancement


class ImpactArea(Enum):
    """Areas of impact for optimization."""
    DATA_ACCURACY = "data_accuracy"
    DATA_COMPLETENESS = "data_completeness"
    DATA_CONSISTENCY = "data_consistency"
    SYSTEM_PERFORMANCE = "system_performance"
    OPERATIONAL_EFFICIENCY = "operational_efficiency"
    COST_REDUCTION = "cost_reduction"
    RISK_MITIGATION = "risk_mitigation"
    USER_EXPERIENCE = "user_experience"


class ImplementationComplexity(Enum):
    """Implementation complexity levels."""
    SIMPLE = "simple"         # < 1 day
    MODERATE = "moderate"     # 1-5 days
    COMPLEX = "complex"       # 1-4 weeks
    VERY_COMPLEX = "very_complex"  # > 1 month


@dataclass
class OptimizationRecommendation:
    """Quality optimization recommendation."""
    recommendation_id: str
    title: str
    description: str
    optimization_type: OptimizationType
    priority: OptimizationPriority
    impact_areas: List[ImpactArea]
    implementation_complexity: ImplementationComplexity
    
    # Impact assessment
    estimated_quality_improvement: float  # 0-1 scale
    estimated_cost_savings: Optional[float] = None
    estimated_risk_reduction: Optional[float] = None
    
    # Implementation details
    implementation_steps: List[str] = field(default_factory=list)
    required_resources: List[str] = field(default_factory=list)
    estimated_effort_hours: Optional[int] = None
    prerequisites: List[str] = field(default_factory=list)
    
    # Technical details
    affected_datasets: List[str] = field(default_factory=list)
    affected_columns: List[str] = field(default_factory=list)
    suggested_tools: List[str] = field(default_factory=list)
    code_examples: List[str] = field(default_factory=list)
    
    # Validation and monitoring
    success_metrics: List[str] = field(default_factory=list)
    monitoring_requirements: List[str] = field(default_factory=list)
    rollback_plan: Optional[str] = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    
    def get_priority_score(self) -> int:
        """Get numerical priority score for sorting."""
        priority_scores = {
            OptimizationPriority.CRITICAL: 5,
            OptimizationPriority.HIGH: 4,
            OptimizationPriority.MEDIUM: 3,
            OptimizationPriority.LOW: 2,
            OptimizationPriority.NICE_TO_HAVE: 1
        }
        return priority_scores.get(self.priority, 0)
    
    def get_roi_estimate(self) -> float:
        """Get estimated return on investment."""
        if self.estimated_effort_hours and self.estimated_effort_hours > 0:
            # Simple ROI calculation: improvement / effort
            return self.estimated_quality_improvement / (self.estimated_effort_hours / 40)  # Normalize to weeks
        return self.estimated_quality_improvement
    
    def get_recommendation_summary(self) -> Dict[str, Any]:
        """Get recommendation summary."""
        return {
            'recommendation_id': self.recommendation_id,
            'title': self.title,
            'description': self.description,
            'optimization_type': self.optimization_type.value,
            'priority': self.priority.value,
            'priority_score': self.get_priority_score(),
            'impact_areas': [area.value for area in self.impact_areas],
            'implementation_complexity': self.implementation_complexity.value,
            'estimated_quality_improvement': round(self.estimated_quality_improvement, 3),
            'estimated_effort_hours': self.estimated_effort_hours,
            'roi_estimate': round(self.get_roi_estimate(), 3),
            'affected_datasets_count': len(self.affected_datasets),
            'category': self.category,
            'created_at': self.created_at.isoformat(),
            'tags': self.tags
        }


@dataclass
class OptimizationPlan:
    """Comprehensive optimization plan."""
    plan_id: str
    dataset_id: str
    plan_name: str
    
    # Recommendations
    recommendations: List[OptimizationRecommendation]
    
    # Plan metrics
    total_estimated_improvement: float
    total_estimated_effort_hours: int
    estimated_completion_weeks: int
    
    # Prioritization
    quick_wins: List[OptimizationRecommendation] = field(default_factory=list)
    high_impact_items: List[OptimizationRecommendation] = field(default_factory=list)
    long_term_initiatives: List[OptimizationRecommendation] = field(default_factory=list)
    
    # Timeline
    implementation_phases: List[Dict[str, Any]] = field(default_factory=list)
    
    # Success criteria
    target_quality_score: float = 0.9
    success_metrics: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def get_plan_summary(self) -> Dict[str, Any]:
        """Get optimization plan summary."""
        return {
            'plan_id': self.plan_id,
            'dataset_id': self.dataset_id,
            'plan_name': self.plan_name,
            'total_recommendations': len(self.recommendations),
            'quick_wins_count': len(self.quick_wins),
            'high_impact_count': len(self.high_impact_items),
            'long_term_count': len(self.long_term_initiatives),
            'total_estimated_improvement': round(self.total_estimated_improvement, 3),
            'total_estimated_effort_hours': self.total_estimated_effort_hours,
            'estimated_completion_weeks': self.estimated_completion_weeks,
            'target_quality_score': self.target_quality_score,
            'implementation_phases_count': len(self.implementation_phases),
            'created_at': self.created_at.isoformat(),
            'last_updated': self.last_updated.isoformat()
        }


class OptimizationStrategy(ABC):
    """Abstract base class for optimization strategies."""
    
    @abstractmethod
    def analyze(self, 
               data_profile: DataQualityProfile,
               quality_score: AdvancedQualityScore,
               historical_data: Optional[List[Dict[str, Any]]] = None) -> List[OptimizationRecommendation]:
        """Analyze and generate optimization recommendations."""
        pass


class CompletenessOptimizationStrategy(OptimizationStrategy):
    """Strategy for optimizing data completeness."""
    
    def analyze(self,
               data_profile: DataQualityProfile,
               quality_score: AdvancedQualityScore,
               historical_data: Optional[List[Dict[str, Any]]] = None) -> List[OptimizationRecommendation]:
        """Analyze completeness issues and generate recommendations."""
        recommendations = []
        
        completeness_metric = quality_score.metrics.get(MetricType.COMPLETENESS)
        if not completeness_metric or completeness_metric.value >= 0.9:
            return recommendations
        
        # Analyze missing value patterns
        if completeness_metric.value < 0.8:
            recommendations.append(OptimizationRecommendation(
                recommendation_id=f"completeness_critical_{data_profile.dataset_id}",
                title="Critical Completeness Issues",
                description=f"Dataset has {(1-completeness_metric.value)*100:.1f}% missing data requiring immediate attention",
                optimization_type=OptimizationType.IMMEDIATE_FIX,
                priority=OptimizationPriority.CRITICAL,
                impact_areas=[ImpactArea.DATA_COMPLETENESS, ImpactArea.DATA_ACCURACY],
                implementation_complexity=ImplementationComplexity.MODERATE,
                estimated_quality_improvement=0.3,
                estimated_effort_hours=40,
                implementation_steps=[
                    "Identify columns with highest missing value rates",
                    "Analyze missing value patterns (MCAR, MAR, MNAR)",
                    "Implement appropriate imputation strategies",
                    "Add validation rules to prevent future missing values",
                    "Monitor completeness metrics daily"
                ],
                affected_datasets=[data_profile.dataset_id],
                success_metrics=["Completeness score > 85%", "Missing value rate < 15%"],
                category="completeness"
            ))
        
        # Column-specific recommendations
        if hasattr(data_profile, 'column_profiles'):
            low_completeness_columns = []
            for column_name, column_profile in data_profile.column_profiles.items():
                if hasattr(column_profile, 'completeness') and column_profile.completeness < 0.7:
                    low_completeness_columns.append(column_name)
            
            if low_completeness_columns:
                recommendations.append(OptimizationRecommendation(
                    recommendation_id=f"column_completeness_{data_profile.dataset_id}",
                    title="Column-Specific Completeness Improvements",
                    description=f"Improve completeness for {len(low_completeness_columns)} columns with high missing rates",
                    optimization_type=OptimizationType.PROCESS_IMPROVEMENT,
                    priority=OptimizationPriority.HIGH,
                    impact_areas=[ImpactArea.DATA_COMPLETENESS],
                    implementation_complexity=ImplementationComplexity.MODERATE,
                    estimated_quality_improvement=0.2,
                    estimated_effort_hours=24,
                    affected_columns=low_completeness_columns,
                    implementation_steps=[
                        f"Focus on columns: {', '.join(low_completeness_columns[:5])}",
                        "Implement column-specific validation rules",
                        "Add required field indicators in data entry forms",
                        "Implement default value strategies where appropriate"
                    ],
                    category="completeness"
                ))
        
        return recommendations


class AccuracyOptimizationStrategy(OptimizationStrategy):
    """Strategy for optimizing data accuracy."""
    
    def analyze(self,
               data_profile: DataQualityProfile,
               quality_score: AdvancedQualityScore,
               historical_data: Optional[List[Dict[str, Any]]] = None) -> List[OptimizationRecommendation]:
        """Analyze accuracy issues and generate recommendations."""
        recommendations = []
        
        accuracy_metric = quality_score.metrics.get(MetricType.ACCURACY)
        if not accuracy_metric or accuracy_metric.value >= 0.9:
            return recommendations
        
        # Data type accuracy issues
        if accuracy_metric.value < 0.7:
            recommendations.append(OptimizationRecommendation(
                recommendation_id=f"accuracy_critical_{data_profile.dataset_id}",
                title="Critical Data Accuracy Issues",
                description="Implement comprehensive data validation and cleansing procedures",
                optimization_type=OptimizationType.IMMEDIATE_FIX,
                priority=OptimizationPriority.CRITICAL,
                impact_areas=[ImpactArea.DATA_ACCURACY, ImpactArea.RISK_MITIGATION],
                implementation_complexity=ImplementationComplexity.COMPLEX,
                estimated_quality_improvement=0.4,
                estimated_effort_hours=80,
                implementation_steps=[
                    "Audit data sources for accuracy issues",
                    "Implement real-time data validation",
                    "Set up data profiling and monitoring",
                    "Create data cleansing workflows",
                    "Establish data stewardship processes"
                ],
                required_resources=["Data Engineer", "Data Steward", "Business Analyst"],
                suggested_tools=["Great Expectations", "Deequ", "Data validation frameworks"],
                category="accuracy"
            ))
        
        # Pattern accuracy
        recommendations.append(OptimizationRecommendation(
            recommendation_id=f"pattern_validation_{data_profile.dataset_id}",
            title="Enhance Pattern Validation",
            description="Implement advanced pattern matching and format validation",
            optimization_type=OptimizationType.PROCESS_IMPROVEMENT,
            priority=OptimizationPriority.HIGH,
            impact_areas=[ImpactArea.DATA_ACCURACY, ImpactArea.DATA_CONSISTENCY],
            implementation_complexity=ImplementationComplexity.MODERATE,
            estimated_quality_improvement=0.2,
            estimated_effort_hours=32,
            implementation_steps=[
                "Identify common data patterns (emails, phone numbers, IDs)",
                "Implement regex-based validation rules",
                "Add format standardization procedures",
                "Create pattern violation alerts"
            ],
            category="accuracy"
        ))
        
        return recommendations


class ConsistencyOptimizationStrategy(OptimizationStrategy):
    """Strategy for optimizing data consistency."""
    
    def analyze(self,
               data_profile: DataQualityProfile,
               quality_score: AdvancedQualityScore,
               historical_data: Optional[List[Dict[str, Any]]] = None) -> List[OptimizationRecommendation]:
        """Analyze consistency issues and generate recommendations."""
        recommendations = []
        
        consistency_metric = quality_score.metrics.get(MetricType.CONSISTENCY)
        if not consistency_metric or consistency_metric.value >= 0.9:
            return recommendations
        
        # Format consistency
        recommendations.append(OptimizationRecommendation(
            recommendation_id=f"format_standardization_{data_profile.dataset_id}",
            title="Data Format Standardization",
            description="Standardize data formats across all columns and sources",
            optimization_type=OptimizationType.PROCESS_IMPROVEMENT,
            priority=OptimizationPriority.MEDIUM,
            impact_areas=[ImpactArea.DATA_CONSISTENCY, ImpactArea.OPERATIONAL_EFFICIENCY],
            implementation_complexity=ImplementationComplexity.MODERATE,
            estimated_quality_improvement=0.25,
            estimated_effort_hours=48,
            implementation_steps=[
                "Define standard formats for each data type",
                "Implement format conversion functions",
                "Create format validation rules",
                "Set up automated format correction"
            ],
            category="consistency"
        ))
        
        # Cross-column consistency
        if consistency_metric.value < 0.7:
            recommendations.append(OptimizationRecommendation(
                recommendation_id=f"cross_column_validation_{data_profile.dataset_id}",
                title="Cross-Column Consistency Validation",
                description="Implement validation rules for related columns",
                optimization_type=OptimizationType.MONITORING_ENHANCEMENT,
                priority=OptimizationPriority.HIGH,
                impact_areas=[ImpactArea.DATA_CONSISTENCY, ImpactArea.DATA_ACCURACY],
                implementation_complexity=ImplementationComplexity.COMPLEX,
                estimated_quality_improvement=0.3,
                estimated_effort_hours=56,
                implementation_steps=[
                    "Identify column relationships and dependencies",
                    "Create business rule validation checks",
                    "Implement referential integrity constraints",
                    "Add cross-column validation alerts"
                ],
                category="consistency"
            ))
        
        return recommendations


@dataclass(frozen=True)
class QualityOptimizationConfig:
    """Configuration for quality optimization service."""
    # Strategy selection
    enable_all_strategies: bool = True
    custom_strategies: List[OptimizationStrategy] = field(default_factory=list)
    
    # Recommendation filtering
    min_quality_improvement: float = 0.05
    max_recommendations_per_category: int = 10
    prioritize_quick_wins: bool = True
    
    # Plan generation
    enable_plan_generation: bool = True
    target_quality_improvement: float = 0.3
    max_plan_duration_weeks: int = 12
    
    # Implementation guidance
    include_code_examples: bool = True
    include_tool_recommendations: bool = True
    generate_implementation_timeline: bool = True
    
    # Monitoring and validation
    include_success_metrics: bool = True
    include_monitoring_requirements: bool = True
    generate_rollback_plans: bool = True


class QualityOptimizationService:
    """Service for quality optimization and improvement recommendations."""
    
    def __init__(self, config: QualityOptimizationConfig = None):
        """Initialize quality optimization service.
        
        Args:
            config: Service configuration
        """
        self.config = config or QualityOptimizationConfig()
        
        # Initialize optimization strategies
        self._strategies = {
            MetricType.COMPLETENESS: CompletenessOptimizationStrategy(),
            MetricType.ACCURACY: AccuracyOptimizationStrategy(),
            MetricType.CONSISTENCY: ConsistencyOptimizationStrategy()
        }
        
        # Add custom strategies
        for strategy in self.config.custom_strategies:
            # Custom strategies would need to specify which metric they target
            pass
        
        # Storage for optimization plans
        self._optimization_plans: Dict[str, OptimizationPlan] = {}
        
        logger.info("Quality Optimization Service initialized")
    
    def generate_optimization_recommendations(self,
                                           dataset_id: str,
                                           data_profile: DataQualityProfile,
                                           quality_score: AdvancedQualityScore,
                                           historical_data: Optional[List[Dict[str, Any]]] = None) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations for a dataset.
        
        Args:
            dataset_id: Dataset identifier
            data_profile: Data quality profile
            quality_score: Advanced quality score
            historical_data: Historical quality data
            
        Returns:
            List of optimization recommendations
        """
        all_recommendations = []
        
        # Apply all applicable strategies
        for metric_type, strategy in self._strategies.items():
            try:
                recommendations = strategy.analyze(data_profile, quality_score, historical_data)
                all_recommendations.extend(recommendations)
            except Exception as e:
                logger.error(f"Error in {metric_type} optimization strategy: {e}")
        
        # Add general recommendations
        general_recommendations = self._generate_general_recommendations(
            dataset_id, data_profile, quality_score
        )
        all_recommendations.extend(general_recommendations)
        
        # Filter recommendations
        filtered_recommendations = self._filter_recommendations(all_recommendations)
        
        # Sort by priority and impact
        sorted_recommendations = self._sort_recommendations(filtered_recommendations)
        
        logger.info(f"Generated {len(sorted_recommendations)} optimization recommendations for {dataset_id}")
        return sorted_recommendations
    
    def create_optimization_plan(self,
                               dataset_id: str,
                               recommendations: List[OptimizationRecommendation],
                               target_quality_score: float = 0.9,
                               max_duration_weeks: int = None) -> OptimizationPlan:
        """Create comprehensive optimization plan.
        
        Args:
            dataset_id: Dataset identifier
            recommendations: List of recommendations
            target_quality_score: Target quality score to achieve
            max_duration_weeks: Maximum plan duration
            
        Returns:
            Optimization plan
        """
        if not self.config.enable_plan_generation:
            raise ValueError("Plan generation is disabled in configuration")
        
        plan_id = f"opt_plan_{dataset_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        max_weeks = max_duration_weeks or self.config.max_plan_duration_weeks
        
        # Categorize recommendations
        quick_wins = []
        high_impact_items = []
        long_term_initiatives = []
        
        for rec in recommendations:
            if (rec.implementation_complexity in [ImplementationComplexity.SIMPLE, ImplementationComplexity.MODERATE] 
                and rec.estimated_quality_improvement >= 0.1):
                quick_wins.append(rec)
            elif rec.estimated_quality_improvement >= 0.2:
                high_impact_items.append(rec)
            else:
                long_term_initiatives.append(rec)
        
        # Calculate plan metrics
        total_improvement = sum(rec.estimated_quality_improvement for rec in recommendations)
        total_effort = sum(rec.estimated_effort_hours or 40 for rec in recommendations)
        estimated_weeks = max(1, min(max_weeks, total_effort // 40))  # Assuming 40 hours/week
        
        # Generate implementation phases
        phases = self._generate_implementation_phases(
            quick_wins, high_impact_items, long_term_initiatives, estimated_weeks
        )
        
        # Create optimization plan
        plan = OptimizationPlan(
            plan_id=plan_id,
            dataset_id=dataset_id,
            plan_name=f"Quality Optimization Plan for {dataset_id}",
            recommendations=recommendations,
            total_estimated_improvement=total_improvement,
            total_estimated_effort_hours=total_effort,
            estimated_completion_weeks=estimated_weeks,
            quick_wins=quick_wins,
            high_impact_items=high_impact_items,
            long_term_initiatives=long_term_initiatives,
            implementation_phases=phases,
            target_quality_score=target_quality_score,
            success_metrics=[
                f"Overall quality score >= {target_quality_score}",
                "All critical issues resolved",
                "Completeness score >= 90%",
                "Accuracy score >= 85%"
            ]
        )
        
        # Store plan
        self._optimization_plans[plan_id] = plan
        
        logger.info(f"Created optimization plan {plan_id} for {dataset_id}")
        return plan
    
    def _generate_general_recommendations(self,
                                        dataset_id: str,
                                        data_profile: DataQualityProfile,
                                        quality_score: AdvancedQualityScore) -> List[OptimizationRecommendation]:
        """Generate general optimization recommendations."""
        recommendations = []
        
        # Monitoring enhancement
        recommendations.append(OptimizationRecommendation(
            recommendation_id=f"monitoring_enhancement_{dataset_id}",
            title="Enhanced Quality Monitoring",
            description="Implement comprehensive data quality monitoring and alerting",
            optimization_type=OptimizationType.MONITORING_ENHANCEMENT,
            priority=OptimizationPriority.MEDIUM,
            impact_areas=[ImpactArea.RISK_MITIGATION, ImpactArea.OPERATIONAL_EFFICIENCY],
            implementation_complexity=ImplementationComplexity.MODERATE,
            estimated_quality_improvement=0.15,
            estimated_effort_hours=32,
            implementation_steps=[
                "Set up automated quality monitoring dashboards",
                "Configure quality metric alerts and thresholds",
                "Implement trend analysis and anomaly detection",
                "Create quality reporting workflows"
            ],
            suggested_tools=["Great Expectations", "Monte Carlo", "Datadog"],
            category="monitoring"
        ))
        
        # Automation opportunities
        if quality_score.overall_score < 0.8:
            recommendations.append(OptimizationRecommendation(
                recommendation_id=f"automation_opportunity_{dataset_id}",
                title="Data Quality Automation",
                description="Automate quality checks and remediation processes",
                optimization_type=OptimizationType.AUTOMATION_OPPORTUNITY,
                priority=OptimizationPriority.HIGH,
                impact_areas=[ImpactArea.OPERATIONAL_EFFICIENCY, ImpactArea.COST_REDUCTION],
                implementation_complexity=ImplementationComplexity.COMPLEX,
                estimated_quality_improvement=0.25,
                estimated_effort_hours=120,
                implementation_steps=[
                    "Identify repetitive quality check processes",
                    "Implement automated data validation pipelines",
                    "Set up automated data cleansing workflows",
                    "Create self-healing data processes"
                ],
                category="automation"
            ))
        
        # Infrastructure improvements
        if len(quality_score.get_critical_metrics()) > 2:
            recommendations.append(OptimizationRecommendation(
                recommendation_id=f"infrastructure_upgrade_{dataset_id}",
                title="Data Infrastructure Improvements",
                description="Upgrade data infrastructure to support better quality management",
                optimization_type=OptimizationType.INFRASTRUCTURE_UPGRADE,
                priority=OptimizationPriority.MEDIUM,
                impact_areas=[ImpactArea.SYSTEM_PERFORMANCE, ImpactArea.OPERATIONAL_EFFICIENCY],
                implementation_complexity=ImplementationComplexity.VERY_COMPLEX,
                estimated_quality_improvement=0.3,
                estimated_effort_hours=200,
                implementation_steps=[
                    "Assess current data infrastructure",
                    "Design improved data architecture",
                    "Implement data quality frameworks",
                    "Migrate to improved infrastructure"
                ],
                category="infrastructure"
            ))
        
        return recommendations
    
    def _filter_recommendations(self, recommendations: List[OptimizationRecommendation]) -> List[OptimizationRecommendation]:
        """Filter recommendations based on configuration."""
        filtered = []
        category_counts = defaultdict(int)
        
        for rec in recommendations:
            # Filter by minimum improvement
            if rec.estimated_quality_improvement < self.config.min_quality_improvement:
                continue
            
            # Filter by category limit
            if category_counts[rec.category] >= self.config.max_recommendations_per_category:
                continue
            
            filtered.append(rec)
            category_counts[rec.category] += 1
        
        return filtered
    
    def _sort_recommendations(self, recommendations: List[OptimizationRecommendation]) -> List[OptimizationRecommendation]:
        """Sort recommendations by priority and impact."""
        def sort_key(rec: OptimizationRecommendation) -> Tuple[int, float, int]:
            priority_score = rec.get_priority_score()
            roi_score = rec.get_roi_estimate()
            complexity_penalty = {
                ImplementationComplexity.SIMPLE: 0,
                ImplementationComplexity.MODERATE: 1,
                ImplementationComplexity.COMPLEX: 2,
                ImplementationComplexity.VERY_COMPLEX: 3
            }.get(rec.implementation_complexity, 2)
            
            return (-priority_score, -roi_score, complexity_penalty)
        
        return sorted(recommendations, key=sort_key)
    
    def _generate_implementation_phases(self,
                                      quick_wins: List[OptimizationRecommendation],
                                      high_impact: List[OptimizationRecommendation],
                                      long_term: List[OptimizationRecommendation],
                                      total_weeks: int) -> List[Dict[str, Any]]:
        """Generate implementation phases for the optimization plan."""
        phases = []
        
        # Phase 1: Quick Wins (First 2-4 weeks)
        if quick_wins:
            phase1_weeks = min(4, total_weeks // 3)
            phases.append({
                'phase_number': 1,
                'phase_name': 'Quick Wins',
                'duration_weeks': phase1_weeks,
                'recommendations': [rec.recommendation_id for rec in quick_wins[:5]],
                'objectives': [
                    'Achieve immediate quality improvements',
                    'Build momentum for larger initiatives',
                    'Demonstrate value of quality improvements'
                ],
                'success_criteria': [
                    'Complete all quick win recommendations',
                    'Achieve 10-15% quality improvement',
                    'No regression in existing quality metrics'
                ]
            })
        
        # Phase 2: High Impact Items (Middle weeks)
        if high_impact:
            phase2_weeks = min(6, total_weeks // 2)
            phases.append({
                'phase_number': 2,
                'phase_name': 'High Impact Improvements',
                'duration_weeks': phase2_weeks,
                'recommendations': [rec.recommendation_id for rec in high_impact[:3]],
                'objectives': [
                    'Address major quality issues',
                    'Implement comprehensive solutions',
                    'Establish sustainable quality processes'
                ],
                'success_criteria': [
                    'Resolve all critical quality issues',
                    'Achieve 20-30% quality improvement',
                    'Implement monitoring and alerting'
                ]
            })
        
        # Phase 3: Long-term Initiatives (Final weeks)
        if long_term:
            phase3_weeks = total_weeks - sum(p['duration_weeks'] for p in phases)
            if phase3_weeks > 0:
                phases.append({
                    'phase_number': 3,
                    'phase_name': 'Strategic Improvements',
                    'duration_weeks': phase3_weeks,
                    'recommendations': [rec.recommendation_id for rec in long_term[:3]],
                    'objectives': [
                        'Implement strategic improvements',
                        'Optimize long-term quality management',
                        'Establish quality excellence'
                    ],
                    'success_criteria': [
                        'Achieve target quality score',
                        'Implement all planned improvements',
                        'Establish ongoing quality governance'
                    ]
                })
        
        return phases
    
    def get_quick_wins(self, recommendations: List[OptimizationRecommendation]) -> List[OptimizationRecommendation]:
        """Get quick win recommendations.
        
        Args:
            recommendations: List of recommendations
            
        Returns:
            Quick win recommendations
        """
        quick_wins = []
        
        for rec in recommendations:
            # Quick wins: Simple/Moderate complexity with good improvement
            if (rec.implementation_complexity in [ImplementationComplexity.SIMPLE, ImplementationComplexity.MODERATE]
                and rec.estimated_quality_improvement >= 0.1
                and rec.priority in [OptimizationPriority.HIGH, OptimizationPriority.CRITICAL]):
                quick_wins.append(rec)
        
        # Sort by ROI
        quick_wins.sort(key=lambda x: x.get_roi_estimate(), reverse=True)
        
        return quick_wins[:5]  # Top 5 quick wins
    
    def estimate_optimization_impact(self,
                                   current_score: float,
                                   recommendations: List[OptimizationRecommendation]) -> Dict[str, Any]:
        """Estimate the impact of implementing recommendations.
        
        Args:
            current_score: Current quality score
            recommendations: List of recommendations
            
        Returns:
            Impact estimation
        """
        # Calculate potential improvement (not simply additive)
        total_improvement = 0
        remaining_improvement_space = 1.0 - current_score
        
        # Sort by priority for cumulative impact calculation
        sorted_recs = sorted(recommendations, key=lambda x: x.get_priority_score(), reverse=True)
        
        for rec in sorted_recs:
            # Each recommendation improves the remaining space
            improvement_contribution = rec.estimated_quality_improvement * remaining_improvement_space
            total_improvement += improvement_contribution
            remaining_improvement_space *= (1 - rec.estimated_quality_improvement)
        
        projected_score = min(1.0, current_score + total_improvement)
        
        # Calculate effort and timeline
        total_effort = sum(rec.estimated_effort_hours or 40 for rec in recommendations)
        estimated_weeks = max(1, total_effort // 40)  # Assuming 40 hours/week
        
        # Calculate cost-benefit
        quick_wins = self.get_quick_wins(recommendations)
        quick_win_improvement = sum(rec.estimated_quality_improvement for rec in quick_wins)
        
        return {
            'current_score': current_score,
            'projected_score': round(projected_score, 3),
            'total_improvement': round(total_improvement, 3),
            'improvement_percentage': round(total_improvement * 100, 1),
            'total_effort_hours': total_effort,
            'estimated_weeks': estimated_weeks,
            'quick_wins_count': len(quick_wins),
            'quick_win_improvement': round(quick_win_improvement, 3),
            'roi_estimate': round(total_improvement / max(estimated_weeks, 1), 3),
            'recommendations_by_priority': {
                priority.value: len([r for r in recommendations if r.priority == priority])
                for priority in OptimizationPriority
            }
        }
    
    def get_optimization_plan(self, plan_id: str) -> Optional[OptimizationPlan]:
        """Get optimization plan by ID.
        
        Args:
            plan_id: Plan identifier
            
        Returns:
            Optimization plan if found
        """
        return self._optimization_plans.get(plan_id)
    
    def list_optimization_plans(self, dataset_id: str = None) -> List[Dict[str, Any]]:
        """List optimization plans.
        
        Args:
            dataset_id: Filter by dataset ID
            
        Returns:
            List of plan summaries
        """
        plans = []
        
        for plan in self._optimization_plans.values():
            if dataset_id and plan.dataset_id != dataset_id:
                continue
            
            plans.append(plan.get_plan_summary())
        
        return plans
    
    def generate_optimization_report(self,
                                   dataset_id: str,
                                   recommendations: List[OptimizationRecommendation],
                                   current_quality_score: float) -> Dict[str, Any]:
        """Generate comprehensive optimization report.
        
        Args:
            dataset_id: Dataset identifier
            recommendations: List of recommendations
            current_quality_score: Current quality score
            
        Returns:
            Optimization report
        """
        # Get impact estimation
        impact_analysis = self.estimate_optimization_impact(current_quality_score, recommendations)
        
        # Get quick wins
        quick_wins = self.get_quick_wins(recommendations)
        
        # Categorize recommendations
        recommendations_by_type = defaultdict(list)
        for rec in recommendations:
            recommendations_by_type[rec.optimization_type.value].append(rec)
        
        recommendations_by_priority = defaultdict(list)
        for rec in recommendations:
            recommendations_by_priority[rec.priority.value].append(rec)
        
        # Generate summary
        report = {
            'dataset_id': dataset_id,
            'report_generated_at': datetime.now().isoformat(),
            'current_quality_assessment': {
                'overall_score': current_quality_score,
                'quality_tier': 'excellent' if current_quality_score >= 0.95 else
                               'good' if current_quality_score >= 0.8 else
                               'fair' if current_quality_score >= 0.6 else
                               'poor' if current_quality_score >= 0.4 else 'critical'
            },
            'optimization_summary': {
                'total_recommendations': len(recommendations),
                'critical_priority_count': len(recommendations_by_priority['critical']),
                'high_priority_count': len(recommendations_by_priority['high']),
                'quick_wins_available': len(quick_wins),
                'estimated_improvement': impact_analysis['total_improvement'],
                'estimated_effort_weeks': impact_analysis['estimated_weeks']
            },
            'impact_analysis': impact_analysis,
            'quick_wins': [rec.get_recommendation_summary() for rec in quick_wins],
            'recommendations_by_type': {
                opt_type: [rec.get_recommendation_summary() for rec in recs]
                for opt_type, recs in recommendations_by_type.items()
            },
            'recommendations_by_priority': {
                priority: [rec.get_recommendation_summary() for rec in recs]
                for priority, recs in recommendations_by_priority.items()
            },
            'implementation_roadmap': {
                'phase_1_quick_wins': [rec.title for rec in quick_wins[:3]],
                'phase_2_high_impact': [rec.title for rec in recommendations_by_priority['high'][:3]],
                'phase_3_strategic': [rec.title for rec in recommendations_by_priority['medium'][:2]]
            },
            'success_metrics': [
                f"Achieve quality score >= {current_quality_score + impact_analysis['total_improvement']:.2f}",
                "Resolve all critical priority recommendations",
                "Implement all quick wins within 4 weeks",
                "Establish ongoing quality monitoring"
            ]
        }
        
        return report