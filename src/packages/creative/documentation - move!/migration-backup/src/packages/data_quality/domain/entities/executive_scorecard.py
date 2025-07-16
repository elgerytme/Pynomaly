"""
Executive Quality Scorecard Entities
Provides C-level visibility into data quality performance with strategic insights.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from uuid import uuid4
import numpy as np


class ScoreCardLevel(str, Enum):
    """Scorecard detail level."""
    EXECUTIVE = "executive"         # C-level overview
    OPERATIONAL = "operational"     # Department-level detail
    TACTICAL = "tactical"          # Team-level granular view


class QualityMaturityStage(str, Enum):
    """Data quality maturity assessment stages."""
    INITIAL = "initial"           # Ad-hoc, reactive quality management
    MANAGED = "managed"           # Basic quality processes in place
    DEFINED = "defined"           # Standardized quality procedures
    QUANTITATIVELY_MANAGED = "quantitatively_managed"  # Metrics-driven quality
    OPTIMIZING = "optimizing"     # Continuous improvement culture


class BusinessImpactSeverity(str, Enum):
    """Business impact severity levels."""
    CRITICAL = "critical"         # Business-critical impact
    HIGH = "high"                # Significant operational impact
    MEDIUM = "medium"            # Moderate business impact
    LOW = "low"                  # Minimal impact
    NEGLIGIBLE = "negligible"    # No material impact


class TrendDirection(str, Enum):
    """Performance trend directions."""
    IMPROVING = "improving"       # Quality metrics improving
    STABLE = "stable"            # Quality metrics stable
    DECLINING = "declining"      # Quality metrics declining
    VOLATILE = "volatile"        # Quality metrics unstable


@dataclass(frozen=True)
class QualityMetricSnapshot:
    """Point-in-time quality metric measurement."""
    metric_name: str
    current_value: float
    target_value: float
    baseline_value: float
    measurement_timestamp: datetime
    data_source: str
    confidence_level: float = 0.95
    
    @property
    def performance_ratio(self) -> float:
        """Calculate performance as ratio to target."""
        if self.target_value == 0:
            return 1.0 if self.current_value == 0 else float('inf')
        return self.current_value / self.target_value
    
    @property
    def improvement_from_baseline(self) -> float:
        """Calculate improvement percentage from baseline."""
        if self.baseline_value == 0:
            return 0.0
        return ((self.current_value - self.baseline_value) / self.baseline_value) * 100


@dataclass(frozen=True)
class BusinessUnit:
    """Business unit organization entity."""
    unit_id: str
    unit_name: str
    division: str
    region: Optional[str] = None
    head_executive: Optional[str] = None
    data_steward: Optional[str] = None
    quality_budget: Optional[float] = None
    employee_count: Optional[int] = None
    revenue_impact: Optional[float] = None


@dataclass(frozen=True)
class QualityKPI:
    """Key Performance Indicator for data quality."""
    kpi_id: str
    name: str
    description: str
    calculation_method: str
    target_value: float
    current_value: float
    trend_direction: TrendDirection
    business_units: List[BusinessUnit] = field(default_factory=list)
    historical_values: List[QualityMetricSnapshot] = field(default_factory=list)
    
    @property
    def achievement_percentage(self) -> float:
        """Calculate KPI achievement as percentage."""
        if self.target_value == 0:
            return 100.0 if self.current_value == 0 else 0.0
        return min(100.0, (self.current_value / self.target_value) * 100)
    
    @property
    def status_indicator(self) -> str:
        """Get visual status indicator."""
        achievement = self.achievement_percentage
        if achievement >= 90:
            return "🟢"  # Green - Excellent
        elif achievement >= 75:
            return "🟡"  # Yellow - Good
        elif achievement >= 60:
            return "🟠"  # Orange - Needs Attention
        else:
            return "🔴"  # Red - Critical


@dataclass(frozen=True)
class FinancialImpact:
    """Financial impact assessment of quality issues."""
    impact_id: str = field(default_factory=lambda: str(uuid4()))
    impact_type: str = "quality_issue"  # quality_issue, improvement_opportunity
    severity: BusinessImpactSeverity = BusinessImpactSeverity.MEDIUM
    
    # Financial metrics
    revenue_impact: float = 0.0
    cost_impact: float = 0.0
    productivity_impact: float = 0.0
    compliance_cost: float = 0.0
    opportunity_cost: float = 0.0
    
    # Risk metrics
    customer_satisfaction_impact: float = 0.0
    reputation_risk_score: float = 0.0
    regulatory_risk_score: float = 0.0
    
    # Temporal aspects
    impact_period_days: int = 30
    time_to_resolution_days: Optional[int] = None
    recurring_frequency: Optional[str] = None  # daily, weekly, monthly, quarterly
    
    # Business context
    affected_business_units: List[str] = field(default_factory=list)
    root_cause_category: Optional[str] = None
    mitigation_actions: List[str] = field(default_factory=list)
    
    @property
    def total_financial_impact(self) -> float:
        """Calculate total financial impact."""
        return (
            abs(self.revenue_impact) + 
            abs(self.cost_impact) + 
            abs(self.productivity_impact) + 
            abs(self.compliance_cost) + 
            abs(self.opportunity_cost)
        )
    
    @property
    def annualized_impact(self) -> float:
        """Calculate annualized financial impact."""
        days_per_year = 365
        if self.recurring_frequency == "daily":
            multiplier = days_per_year
        elif self.recurring_frequency == "weekly":
            multiplier = 52
        elif self.recurring_frequency == "monthly":
            multiplier = 12
        elif self.recurring_frequency == "quarterly":
            multiplier = 4
        else:
            multiplier = days_per_year / max(self.impact_period_days, 1)
        
        return self.total_financial_impact * multiplier


@dataclass(frozen=True)
class QualityROIAnalysis:
    """Return on Investment analysis for quality initiatives."""
    analysis_id: str = field(default_factory=lambda: str(uuid4()))
    initiative_name: str = ""
    
    # Investment metrics
    initial_investment: float = 0.0
    ongoing_operational_cost: float = 0.0
    technology_cost: float = 0.0
    training_cost: float = 0.0
    
    # Return metrics
    cost_savings: float = 0.0
    revenue_improvement: float = 0.0
    productivity_gains: float = 0.0
    risk_reduction_value: float = 0.0
    
    # Time metrics
    investment_period_months: int = 12
    payback_period_months: Optional[int] = None
    
    @property
    def total_investment(self) -> float:
        """Calculate total investment cost."""
        return (
            self.initial_investment + 
            self.ongoing_operational_cost + 
            self.technology_cost + 
            self.training_cost
        )
    
    @property
    def total_returns(self) -> float:
        """Calculate total returns."""
        return (
            self.cost_savings + 
            self.revenue_improvement + 
            self.productivity_gains + 
            self.risk_reduction_value
        )
    
    @property
    def roi_percentage(self) -> float:
        """Calculate ROI as percentage."""
        if self.total_investment == 0:
            return 0.0
        return ((self.total_returns - self.total_investment) / self.total_investment) * 100
    
    @property
    def net_present_value(self) -> float:
        """Calculate simple NPV (without discount rate)."""
        return self.total_returns - self.total_investment


@dataclass(frozen=True)
class IndustryBenchmark:
    """Industry benchmark comparison data."""
    benchmark_id: str = field(default_factory=lambda: str(uuid4()))
    industry_sector: str = ""
    company_size_category: str = ""  # startup, small, medium, large, enterprise
    region: str = ""
    
    # Benchmark metrics
    median_quality_score: float = 0.0
    top_quartile_score: float = 0.0
    bottom_quartile_score: float = 0.0
    industry_average_score: float = 0.0
    
    # Investment benchmarks
    median_quality_investment_percentage: float = 0.0
    top_performer_investment_percentage: float = 0.0
    
    # Maturity benchmarks
    typical_maturity_stage: QualityMaturityStage = QualityMaturityStage.INITIAL
    leading_practices: List[str] = field(default_factory=list)
    
    # Performance metrics
    benchmark_timestamp: datetime = field(default_factory=datetime.now)
    data_source: str = "industry_survey"
    sample_size: int = 0
    confidence_interval: float = 0.95


@dataclass
class ExecutiveScorecard:
    """Executive-level data quality scorecard."""
    
    scorecard_id: str = field(default_factory=lambda: str(uuid4()))
    organization_name: str = ""
    reporting_period: str = ""
    scorecard_level: ScoreCardLevel = ScoreCardLevel.EXECUTIVE
    
    # Core quality metrics
    overall_quality_score: float = 0.0
    quality_kpis: List[QualityKPI] = field(default_factory=list)
    quality_trends: Dict[str, TrendDirection] = field(default_factory=dict)
    
    # Maturity assessment
    current_maturity_stage: QualityMaturityStage = QualityMaturityStage.INITIAL
    target_maturity_stage: QualityMaturityStage = QualityMaturityStage.MANAGED
    maturity_progression_score: float = 0.0
    
    # Business impact
    financial_impacts: List[FinancialImpact] = field(default_factory=list)
    roi_analyses: List[QualityROIAnalysis] = field(default_factory=list)
    
    # Benchmarking
    industry_benchmarks: List[IndustryBenchmark] = field(default_factory=list)
    peer_comparisons: Dict[str, float] = field(default_factory=dict)
    
    # Business unit performance
    business_unit_scores: Dict[str, float] = field(default_factory=dict)
    best_performing_units: List[str] = field(default_factory=list)
    underperforming_units: List[str] = field(default_factory=list)
    
    # Strategic insights
    top_quality_risks: List[str] = field(default_factory=list)
    improvement_opportunities: List[str] = field(default_factory=list)
    strategic_recommendations: List[str] = field(default_factory=list)
    
    # Metadata
    generated_timestamp: datetime = field(default_factory=datetime.now)
    data_freshness_hours: float = 0.0
    coverage_percentage: float = 100.0
    confidence_score: float = 0.0
    
    def get_executive_summary(self) -> Dict[str, Any]:
        """Generate executive summary for C-level consumption."""
        total_financial_impact = sum(fi.total_financial_impact for fi in self.financial_impacts)
        total_roi = sum(roi.roi_percentage for roi in self.roi_analyses) / max(len(self.roi_analyses), 1)
        
        critical_risks = len([fi for fi in self.financial_impacts 
                            if fi.severity == BusinessImpactSeverity.CRITICAL])
        
        return {
            "overall_quality_score": f"{self.overall_quality_score:.1f}%",
            "maturity_stage": self.current_maturity_stage.value.title(),
            "total_financial_impact": f"${total_financial_impact:,.0f}",
            "quality_roi": f"{total_roi:.1f}%",
            "critical_risks": critical_risks,
            "top_risk": self.top_quality_risks[0] if self.top_quality_risks else "None identified",
            "key_opportunity": self.improvement_opportunities[0] if self.improvement_opportunities else "Assessment pending",
            "trend_summary": self._get_trend_summary(),
            "benchmark_position": self._get_benchmark_position()
        }
    
    def _get_trend_summary(self) -> str:
        """Summarize quality trends."""
        if not self.quality_trends:
            return "Trend data pending"
        
        improving = sum(1 for trend in self.quality_trends.values() 
                       if trend == TrendDirection.IMPROVING)
        declining = sum(1 for trend in self.quality_trends.values() 
                       if trend == TrendDirection.DECLINING)
        total = len(self.quality_trends)
        
        if improving > declining:
            return f"Improving ({improving}/{total} metrics)"
        elif declining > improving:
            return f"Declining ({declining}/{total} metrics)"
        else:
            return f"Mixed ({total} metrics tracked)"
    
    def _get_benchmark_position(self) -> str:
        """Determine position relative to industry benchmarks."""
        if not self.industry_benchmarks:
            return "Benchmarking in progress"
        
        # Use first benchmark for simplicity
        benchmark = self.industry_benchmarks[0]
        
        if self.overall_quality_score >= benchmark.top_quartile_score:
            return "Top quartile performance"
        elif self.overall_quality_score >= benchmark.median_quality_score:
            return "Above industry median"
        elif self.overall_quality_score >= benchmark.bottom_quartile_score:
            return "Below industry median"
        else:
            return "Bottom quartile - improvement needed"
    
    def calculate_quality_health_index(self) -> float:
        """Calculate overall quality health index (0-100)."""
        if not self.quality_kpis:
            return 0.0
        
        # Weight different aspects
        kpi_score = np.mean([kpi.achievement_percentage for kpi in self.quality_kpis])
        maturity_score = self._maturity_to_score(self.current_maturity_stage) * 20
        trend_score = self._calculate_trend_score() * 25
        risk_score = max(0, 100 - len([fi for fi in self.financial_impacts 
                                     if fi.severity in [BusinessImpactSeverity.CRITICAL, 
                                                       BusinessImpactSeverity.HIGH]]) * 10)
        
        # Weighted average
        weights = [0.4, 0.2, 0.2, 0.2]  # KPIs, maturity, trends, risks
        scores = [kpi_score, maturity_score, trend_score, risk_score]
        
        return np.average(scores, weights=weights)
    
    def _maturity_to_score(self, stage: QualityMaturityStage) -> float:
        """Convert maturity stage to numeric score."""
        stage_scores = {
            QualityMaturityStage.INITIAL: 1.0,
            QualityMaturityStage.MANAGED: 2.0,
            QualityMaturityStage.DEFINED: 3.0,
            QualityMaturityStage.QUANTITATIVELY_MANAGED: 4.0,
            QualityMaturityStage.OPTIMIZING: 5.0
        }
        return stage_scores.get(stage, 1.0)
    
    def _calculate_trend_score(self) -> float:
        """Calculate trend performance score."""
        if not self.quality_trends:
            return 0.0
        
        trend_weights = {
            TrendDirection.IMPROVING: 1.0,
            TrendDirection.STABLE: 0.7,
            TrendDirection.DECLINING: 0.2,
            TrendDirection.VOLATILE: 0.4
        }
        
        total_weight = sum(trend_weights[trend] for trend in self.quality_trends.values())
        return (total_weight / len(self.quality_trends)) * 100