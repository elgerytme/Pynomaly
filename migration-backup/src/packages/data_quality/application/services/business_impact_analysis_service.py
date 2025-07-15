"""
Business Impact Analysis Service
Calculates financial impacts of data quality issues and ROI of quality initiatives.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
import math

from ...domain.entities.executive_scorecard import (
    FinancialImpact, QualityROIAnalysis, BusinessImpactSeverity,
    BusinessUnit, QualityMaturityStage
)
from ...domain.entities.data_quality_assessment import QualityAssessment
from ...domain.entities.quality_rule import QualityRule


@dataclass
class ImpactCalculationConfig:
    """Configuration for impact calculations."""
    
    # Revenue impact factors
    customer_churn_cost_multiplier: float = 5.0  # Cost to acquire vs retain customer
    revenue_per_customer_monthly: float = 1000.0
    customer_lifetime_value: float = 12000.0
    
    # Operational cost factors
    manual_correction_hourly_cost: float = 75.0
    automated_correction_cost_per_record: float = 0.05
    system_downtime_hourly_cost: float = 10000.0
    
    # Productivity factors
    decision_delay_cost_per_hour: float = 500.0
    rework_cost_multiplier: float = 3.0  # Cost of rework vs initial work
    employee_hourly_rate: float = 85.0
    
    # Compliance and risk factors
    regulatory_fine_base_amount: float = 50000.0
    reputation_damage_revenue_impact: float = 0.02  # 2% of annual revenue
    audit_cost_per_finding: float = 5000.0
    
    # Quality initiative cost factors
    tool_licensing_cost_per_user_monthly: float = 100.0
    training_cost_per_employee: float = 2000.0
    consultant_daily_rate: float = 2500.0


@dataclass
class BusinessMetrics:
    """Business metrics for impact calculations."""
    annual_revenue: float = 100_000_000.0  # $100M default
    total_employees: int = 1000
    customer_count: int = 10000
    data_volume_gb: float = 1000.0
    transaction_volume_daily: int = 100000
    system_uptime_target: float = 0.999  # 99.9%
    
    # Industry-specific factors
    industry_sector: str = "technology"
    regulatory_environment: str = "standard"  # standard, high, critical
    competitive_pressure: str = "medium"  # low, medium, high


class BusinessImpactAnalysisService:
    """
    Service for analyzing business impact of data quality issues and initiatives.
    """
    
    def __init__(self, config: ImpactCalculationConfig = None, 
                 business_metrics: BusinessMetrics = None):
        self.config = config or ImpactCalculationConfig()
        self.business_metrics = business_metrics or BusinessMetrics()
        self.logger = logging.getLogger(__name__)
        
        # Impact calculation models
        self._impact_models = self._initialize_impact_models()
        
        # Historical data for trend analysis
        self._historical_impacts: List[FinancialImpact] = []
        self._baseline_metrics: Dict[str, float] = {}
    
    def _initialize_impact_models(self) -> Dict[str, Any]:
        """Initialize impact calculation models."""
        return {
            "customer_impact": self._calculate_customer_impact,
            "operational_impact": self._calculate_operational_impact,
            "productivity_impact": self._calculate_productivity_impact,
            "compliance_impact": self._calculate_compliance_impact,
            "reputation_impact": self._calculate_reputation_impact,
            "opportunity_cost": self._calculate_opportunity_cost
        }
    
    async def analyze_quality_issue_impact(
        self,
        quality_assessment: QualityAssessment,
        affected_data_volume: float,
        business_context: Dict[str, Any] = None
    ) -> FinancialImpact:
        """
        Analyze the business impact of a data quality issue.
        
        Args:
            quality_assessment: Quality assessment results
            affected_data_volume: Volume of data affected (GB or record count)
            business_context: Additional business context
            
        Returns:
            Financial impact analysis
        """
        try:
            self.logger.info("Starting business impact analysis for quality issue")
            
            business_context = business_context or {}
            
            # Determine impact severity based on quality scores
            severity = self._determine_impact_severity(quality_assessment)
            
            # Calculate different types of financial impact
            revenue_impact = await self._calculate_revenue_impact(
                quality_assessment, affected_data_volume, business_context
            )
            
            cost_impact = await self._calculate_cost_impact(
                quality_assessment, affected_data_volume, business_context
            )
            
            productivity_impact = await self._calculate_productivity_impact_detailed(
                quality_assessment, affected_data_volume, business_context
            )
            
            compliance_cost = await self._calculate_compliance_cost_detailed(
                quality_assessment, business_context
            )
            
            opportunity_cost = await self._calculate_opportunity_cost_detailed(
                quality_assessment, business_context
            )
            
            # Calculate risk scores
            customer_impact = self._calculate_customer_satisfaction_impact(quality_assessment)
            reputation_risk = self._calculate_reputation_risk_score(quality_assessment, business_context)
            regulatory_risk = self._calculate_regulatory_risk_score(quality_assessment, business_context)
            
            # Determine affected business units
            affected_units = self._identify_affected_business_units(quality_assessment, business_context)
            
            # Create financial impact entity
            impact = FinancialImpact(
                severity=severity,
                revenue_impact=revenue_impact,
                cost_impact=cost_impact,
                productivity_impact=productivity_impact,
                compliance_cost=compliance_cost,
                opportunity_cost=opportunity_cost,
                customer_satisfaction_impact=customer_impact,
                reputation_risk_score=reputation_risk,
                regulatory_risk_score=regulatory_risk,
                affected_business_units=affected_units,
                root_cause_category=self._categorize_root_cause(quality_assessment),
                mitigation_actions=self._suggest_mitigation_actions(quality_assessment)
            )
            
            # Store for historical analysis
            self._historical_impacts.append(impact)
            
            self.logger.info(f"Impact analysis completed: ${impact.total_financial_impact:,.0f} total impact")
            
            return impact
            
        except Exception as e:
            self.logger.error(f"Error in business impact analysis: {str(e)}")
            raise
    
    async def calculate_quality_initiative_roi(
        self,
        initiative_description: str,
        implementation_plan: Dict[str, Any],
        expected_improvements: Dict[str, float],
        timeline_months: int = 12
    ) -> QualityROIAnalysis:
        """
        Calculate ROI for a data quality initiative.
        
        Args:
            initiative_description: Description of the quality initiative
            implementation_plan: Implementation details and costs
            expected_improvements: Expected quality improvements
            timeline_months: Timeline for ROI calculation
            
        Returns:
            ROI analysis results
        """
        try:
            self.logger.info(f"Calculating ROI for initiative: {initiative_description}")
            
            # Calculate investment costs
            initial_investment = self._calculate_initial_investment(implementation_plan)
            ongoing_operational_cost = self._calculate_ongoing_costs(implementation_plan, timeline_months)
            technology_cost = self._calculate_technology_costs(implementation_plan, timeline_months)
            training_cost = self._calculate_training_costs(implementation_plan)
            
            # Calculate expected returns
            cost_savings = await self._calculate_expected_cost_savings(expected_improvements, timeline_months)
            revenue_improvement = await self._calculate_revenue_improvements(expected_improvements, timeline_months)
            productivity_gains = await self._calculate_productivity_gains(expected_improvements, timeline_months)
            risk_reduction_value = await self._calculate_risk_reduction_value(expected_improvements, timeline_months)
            
            # Calculate payback period
            payback_period = self._calculate_payback_period(
                initial_investment + technology_cost + training_cost,
                (cost_savings + revenue_improvement + productivity_gains) / timeline_months
            )
            
            roi_analysis = QualityROIAnalysis(
                initiative_name=initiative_description,
                initial_investment=initial_investment,
                ongoing_operational_cost=ongoing_operational_cost,
                technology_cost=technology_cost,
                training_cost=training_cost,
                cost_savings=cost_savings,
                revenue_improvement=revenue_improvement,
                productivity_gains=productivity_gains,
                risk_reduction_value=risk_reduction_value,
                investment_period_months=timeline_months,
                payback_period_months=payback_period
            )
            
            self.logger.info(f"ROI calculated: {roi_analysis.roi_percentage:.1f}% over {timeline_months} months")
            
            return roi_analysis
            
        except Exception as e:
            self.logger.error(f"Error calculating ROI: {str(e)}")
            raise
    
    async def analyze_business_unit_performance(
        self,
        business_units: List[BusinessUnit],
        quality_assessments: Dict[str, QualityAssessment]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze quality performance and impact across business units.
        """
        results = {}
        
        for unit in business_units:
            if unit.unit_id not in quality_assessments:
                continue
                
            assessment = quality_assessments[unit.unit_id]
            
            # Calculate unit-specific impact
            impact = await self.analyze_quality_issue_impact(
                assessment,
                affected_data_volume=1000.0,  # Default volume
                business_context={
                    "business_unit": unit.unit_name,
                    "employee_count": unit.employee_count,
                    "revenue_impact_factor": unit.revenue_impact or 1.0
                }
            )
            
            # Calculate unit performance score
            performance_score = self._calculate_unit_performance_score(assessment, impact, unit)
            
            results[unit.unit_id] = {
                "unit_name": unit.unit_name,
                "quality_score": assessment.overall_score,
                "financial_impact": impact.total_financial_impact,
                "performance_score": performance_score,
                "primary_issues": self._identify_primary_issues(assessment),
                "improvement_potential": self._calculate_improvement_potential(assessment, unit),
                "benchmark_position": self._calculate_benchmark_position(assessment)
            }
        
        return results
    
    def _determine_impact_severity(self, assessment: QualityAssessment) -> BusinessImpactSeverity:
        """Determine impact severity based on quality assessment."""
        if assessment.overall_score < 0.5:
            return BusinessImpactSeverity.CRITICAL
        elif assessment.overall_score < 0.7:
            return BusinessImpactSeverity.HIGH
        elif assessment.overall_score < 0.85:
            return BusinessImpactSeverity.MEDIUM
        elif assessment.overall_score < 0.95:
            return BusinessImpactSeverity.LOW
        else:
            return BusinessImpactSeverity.NEGLIGIBLE
    
    async def _calculate_revenue_impact(
        self,
        assessment: QualityAssessment,
        affected_volume: float,
        context: Dict[str, Any]
    ) -> float:
        """Calculate revenue impact from quality issues."""
        
        # Customer churn impact
        churn_factor = max(0, 1 - assessment.overall_score)
        estimated_customer_loss = self.business_metrics.customer_count * churn_factor * 0.1  # 10% max
        churn_revenue_impact = estimated_customer_loss * self.config.customer_lifetime_value
        
        # Transaction failure impact
        transaction_failure_rate = max(0, 1 - assessment.accuracy_score)
        daily_transaction_loss = (self.business_metrics.transaction_volume_daily * 
                                transaction_failure_rate * 0.05)  # 5% max failure rate
        transaction_revenue_impact = (daily_transaction_loss * 30 *  # Monthly
                                    self.config.revenue_per_customer_monthly / 
                                    self.business_metrics.transaction_volume_daily * 
                                    self.business_metrics.customer_count)
        
        # Decision delay impact
        completeness_gap = max(0, 0.95 - assessment.completeness_score)
        decision_delays = completeness_gap * 40  # Hours per month
        decision_delay_impact = decision_delays * self.config.decision_delay_cost_per_hour
        
        return churn_revenue_impact + transaction_revenue_impact + decision_delay_impact
    
    async def _calculate_cost_impact(
        self,
        assessment: QualityAssessment,
        affected_volume: float,
        context: Dict[str, Any]
    ) -> float:
        """Calculate operational cost impact."""
        
        # Data correction costs
        error_rate = max(0, 1 - assessment.accuracy_score)
        records_needing_correction = affected_volume * 1000 * error_rate  # Assume 1000 records per GB
        manual_correction_cost = (records_needing_correction * 0.1 *  # 10% require manual correction
                                self.config.manual_correction_hourly_cost / 60)  # Assume 1 minute per record
        
        # System maintenance costs
        reliability_issues = max(0, 0.95 - assessment.consistency_score)
        maintenance_overhead = reliability_issues * self.business_metrics.annual_revenue * 0.001  # 0.1% of revenue
        
        # Audit and compliance costs
        compliance_gap = max(0, 0.9 - assessment.validity_score)
        audit_costs = compliance_gap * self.config.audit_cost_per_finding * 5  # Average 5 findings
        
        return manual_correction_cost + maintenance_overhead + audit_costs
    
    async def _calculate_productivity_impact_detailed(
        self,
        assessment: QualityAssessment,
        affected_volume: float,
        context: Dict[str, Any]
    ) -> float:
        """Calculate detailed productivity impact."""
        
        # Time spent on data quality issues
        quality_gap = max(0, 0.9 - assessment.overall_score)
        hours_lost_monthly = (self.business_metrics.total_employees * 0.3 *  # 30% of employees affected
                            quality_gap * 8)  # Hours lost per month per employee
        
        productivity_cost = hours_lost_monthly * self.config.employee_hourly_rate
        
        # Rework costs
        rework_factor = max(0, 1 - assessment.consistency_score)
        rework_cost = productivity_cost * rework_factor * self.config.rework_cost_multiplier
        
        return productivity_cost + rework_cost
    
    async def _calculate_compliance_cost_detailed(
        self,
        assessment: QualityAssessment,
        context: Dict[str, Any]
    ) -> float:
        """Calculate compliance-related costs."""
        
        # Regulatory risk factor
        regulatory_multiplier = {
            "standard": 1.0,
            "high": 2.5,
            "critical": 5.0
        }.get(self.business_metrics.regulatory_environment, 1.0)
        
        # Base compliance gap
        compliance_gap = max(0, 0.95 - assessment.validity_score)
        
        # Fine risk calculation
        fine_probability = compliance_gap * 0.2  # 20% max probability
        expected_fine = (fine_probability * 
                        self.config.regulatory_fine_base_amount * 
                        regulatory_multiplier)
        
        # Additional compliance costs
        compliance_overhead = compliance_gap * self.business_metrics.annual_revenue * 0.0005
        
        return expected_fine + compliance_overhead
    
    async def _calculate_opportunity_cost_detailed(
        self,
        assessment: QualityAssessment,
        context: Dict[str, Any]
    ) -> float:
        """Calculate opportunity costs from quality issues."""
        
        # Innovation delay costs
        data_readiness = assessment.overall_score
        innovation_delay_factor = max(0, 0.9 - data_readiness)
        innovation_opportunity_cost = (innovation_delay_factor * 
                                     self.business_metrics.annual_revenue * 0.02)  # 2% of revenue
        
        # Market opportunity costs
        competitive_multiplier = {"low": 0.5, "medium": 1.0, "high": 2.0}.get(
            self.business_metrics.competitive_pressure, 1.0
        )
        
        market_opportunity_cost = (innovation_opportunity_cost * 
                                 competitive_multiplier * 0.5)
        
        return innovation_opportunity_cost + market_opportunity_cost
    
    def _calculate_customer_satisfaction_impact(self, assessment: QualityAssessment) -> float:
        """Calculate customer satisfaction impact score (0-100)."""
        # Data quality directly impacts customer experience
        quality_impact = max(0, 0.95 - assessment.overall_score) * 100
        
        # Accuracy issues have highest customer impact
        accuracy_impact = max(0, 0.98 - assessment.accuracy_score) * 150
        
        # Timeliness issues also affect satisfaction
        timeliness_impact = max(0, 0.9 - assessment.timeliness_score) * 75
        
        return min(100, quality_impact + accuracy_impact + timeliness_impact)
    
    def _calculate_reputation_risk_score(
        self,
        assessment: QualityAssessment,
        context: Dict[str, Any]
    ) -> float:
        """Calculate reputation risk score (0-100)."""
        
        # Base reputation risk from quality issues
        base_risk = max(0, 0.85 - assessment.overall_score) * 100
        
        # Public-facing data issues have higher reputation impact
        public_facing_multiplier = context.get("public_facing", False) and 1.5 or 1.0
        
        # Industry visibility factor
        industry_visibility = {"technology": 1.2, "financial": 1.5, "healthcare": 1.3}.get(
            self.business_metrics.industry_sector, 1.0
        )
        
        return min(100, base_risk * public_facing_multiplier * industry_visibility)
    
    def _calculate_regulatory_risk_score(
        self,
        assessment: QualityAssessment,
        context: Dict[str, Any]
    ) -> float:
        """Calculate regulatory risk score (0-100)."""
        
        # Base regulatory risk
        compliance_gap = max(0, 0.95 - assessment.validity_score)
        base_risk = compliance_gap * 100
        
        # Regulatory environment multiplier
        environment_multiplier = {
            "standard": 1.0,
            "high": 2.0,
            "critical": 3.0
        }.get(self.business_metrics.regulatory_environment, 1.0)
        
        return min(100, base_risk * environment_multiplier)
    
    def _identify_affected_business_units(
        self,
        assessment: QualityAssessment,
        context: Dict[str, Any]
    ) -> List[str]:
        """Identify which business units are affected by quality issues."""
        
        affected_units = []
        
        # If specific unit mentioned in context
        if "business_unit" in context:
            affected_units.append(context["business_unit"])
        
        # Common units affected by quality issues
        if assessment.completeness_score < 0.8:
            affected_units.extend(["Analytics", "Reporting"])
        
        if assessment.accuracy_score < 0.8:
            affected_units.extend(["Customer Service", "Sales"])
        
        if assessment.consistency_score < 0.8:
            affected_units.extend(["Operations", "Finance"])
        
        if assessment.validity_score < 0.8:
            affected_units.extend(["Compliance", "Risk Management"])
        
        return list(set(affected_units))  # Remove duplicates
    
    def _categorize_root_cause(self, assessment: QualityAssessment) -> str:
        """Categorize the root cause of quality issues."""
        
        # Identify primary quality dimension with lowest score
        scores = {
            "completeness": assessment.completeness_score,
            "accuracy": assessment.accuracy_score,
            "consistency": assessment.consistency_score,
            "validity": assessment.validity_score,
            "uniqueness": assessment.uniqueness_score,
            "timeliness": assessment.timeliness_score
        }
        
        lowest_dimension = min(scores.items(), key=lambda x: x[1])[0]
        
        root_cause_mapping = {
            "completeness": "data_collection_gaps",
            "accuracy": "data_entry_errors",
            "consistency": "system_integration_issues",
            "validity": "business_rule_violations",
            "uniqueness": "duplicate_data_management",
            "timeliness": "data_processing_delays"
        }
        
        return root_cause_mapping.get(lowest_dimension, "unknown")
    
    def _suggest_mitigation_actions(self, assessment: QualityAssessment) -> List[str]:
        """Suggest mitigation actions based on quality assessment."""
        
        actions = []
        
        if assessment.completeness_score < 0.8:
            actions.append("Implement mandatory field validation")
            actions.append("Review data collection processes")
        
        if assessment.accuracy_score < 0.8:
            actions.append("Enhance data validation rules")
            actions.append("Implement automated data verification")
        
        if assessment.consistency_score < 0.8:
            actions.append("Standardize data formats across systems")
            actions.append("Implement master data management")
        
        if assessment.validity_score < 0.8:
            actions.append("Update business rules and constraints")
            actions.append("Implement real-time validation")
        
        if assessment.uniqueness_score < 0.8:
            actions.append("Implement duplicate detection and resolution")
            actions.append("Establish data deduplication processes")
        
        if assessment.timeliness_score < 0.8:
            actions.append("Optimize data processing pipelines")
            actions.append("Implement real-time data streaming")
        
        return actions
    
    # ROI Calculation Methods
    
    def _calculate_initial_investment(self, plan: Dict[str, Any]) -> float:
        """Calculate initial investment costs."""
        base_cost = plan.get("base_implementation_cost", 50000.0)
        consulting_days = plan.get("consulting_days", 20)
        consulting_cost = consulting_days * self.config.consultant_daily_rate
        
        infrastructure_cost = plan.get("infrastructure_cost", 25000.0)
        
        return base_cost + consulting_cost + infrastructure_cost
    
    def _calculate_ongoing_costs(self, plan: Dict[str, Any], months: int) -> float:
        """Calculate ongoing operational costs."""
        monthly_cost = plan.get("monthly_operational_cost", 5000.0)
        return monthly_cost * months
    
    def _calculate_technology_costs(self, plan: Dict[str, Any], months: int) -> float:
        """Calculate technology and licensing costs."""
        users = plan.get("user_count", 50)
        monthly_licensing = users * self.config.tool_licensing_cost_per_user_monthly
        
        cloud_costs = plan.get("monthly_cloud_cost", 2000.0)
        
        return (monthly_licensing + cloud_costs) * months
    
    def _calculate_training_costs(self, plan: Dict[str, Any]) -> float:
        """Calculate training costs."""
        employees_to_train = plan.get("employees_to_train", 100)
        return employees_to_train * self.config.training_cost_per_employee
    
    async def _calculate_expected_cost_savings(
        self,
        improvements: Dict[str, float],
        months: int
    ) -> float:
        """Calculate expected cost savings from quality improvements."""
        
        # Reduction in manual correction costs
        accuracy_improvement = improvements.get("accuracy_improvement", 0.0)
        correction_savings = (accuracy_improvement * 
                            self.business_metrics.total_employees * 0.1 *  # 10% time saved
                            self.config.employee_hourly_rate * 20 * months)  # 20 hours/month
        
        # Reduction in system maintenance costs
        consistency_improvement = improvements.get("consistency_improvement", 0.0)
        maintenance_savings = (consistency_improvement * 
                             self.business_metrics.annual_revenue * 0.001 * 
                             months / 12)
        
        return correction_savings + maintenance_savings
    
    async def _calculate_revenue_improvements(
        self,
        improvements: Dict[str, float],
        months: int
    ) -> float:
        """Calculate expected revenue improvements."""
        
        # Customer retention improvement
        overall_improvement = improvements.get("overall_quality_improvement", 0.0)
        retention_improvement = overall_improvement * 0.5  # 50% translates to retention
        
        retained_customers = (self.business_metrics.customer_count * 
                            retention_improvement * 0.05)  # 5% churn reduction
        
        revenue_improvement = (retained_customers * 
                             self.config.revenue_per_customer_monthly * months)
        
        return revenue_improvement
    
    async def _calculate_productivity_gains(
        self,
        improvements: Dict[str, float],
        months: int
    ) -> float:
        """Calculate productivity gains from quality improvements."""
        
        overall_improvement = improvements.get("overall_quality_improvement", 0.0)
        
        # Time savings from better data quality
        time_savings_hours = (self.business_metrics.total_employees * 0.3 *  # 30% affected
                            overall_improvement * 4 * months)  # 4 hours/month saved
        
        productivity_value = time_savings_hours * self.config.employee_hourly_rate
        
        return productivity_value
    
    async def _calculate_risk_reduction_value(
        self,
        improvements: Dict[str, float],
        months: int
    ) -> float:
        """Calculate value from risk reduction."""
        
        compliance_improvement = improvements.get("compliance_improvement", 0.0)
        
        # Reduced regulatory risk
        risk_reduction = (compliance_improvement * 
                        self.config.regulatory_fine_base_amount * 0.2)  # 20% risk reduction
        
        # Reduced reputation risk
        reputation_value = (compliance_improvement * 
                          self.business_metrics.annual_revenue * 
                          self.config.reputation_damage_revenue_impact * 
                          months / 12)
        
        return risk_reduction + reputation_value
    
    def _calculate_payback_period(self, total_investment: float, monthly_benefits: float) -> Optional[int]:
        """Calculate payback period in months."""
        if monthly_benefits <= 0:
            return None
        
        return math.ceil(total_investment / monthly_benefits)
    
    # Business Unit Analysis Methods
    
    def _calculate_unit_performance_score(
        self,
        assessment: QualityAssessment,
        impact: FinancialImpact,
        unit: BusinessUnit
    ) -> float:
        """Calculate overall performance score for business unit."""
        
        # Quality score (40% weight)
        quality_component = assessment.overall_score * 40
        
        # Financial impact score (30% weight) - inverse relationship
        max_impact = 1000000  # $1M reference point
        impact_score = max(0, 100 - (impact.total_financial_impact / max_impact * 100))
        impact_component = impact_score * 0.3
        
        # Maturity score (20% weight) - based on unit characteristics
        maturity_component = self._estimate_unit_maturity_score(unit) * 20
        
        # Improvement trend (10% weight) - placeholder
        trend_component = 70 * 0.1  # Neutral trend
        
        return quality_component + impact_component + maturity_component + trend_component
    
    def _estimate_unit_maturity_score(self, unit: BusinessUnit) -> float:
        """Estimate data quality maturity score for business unit."""
        base_score = 50.0  # Base maturity
        
        # Larger units typically have better processes
        if unit.employee_count and unit.employee_count > 100:
            base_score += 20
        
        # Units with dedicated stewards score higher
        if unit.data_steward:
            base_score += 15
        
        # Units with quality budget score higher
        if unit.quality_budget and unit.quality_budget > 0:
            base_score += 15
        
        return min(100.0, base_score)
    
    def _identify_primary_issues(self, assessment: QualityAssessment) -> List[str]:
        """Identify primary quality issues for business unit."""
        issues = []
        
        if assessment.completeness_score < 0.8:
            issues.append("Data Completeness")
        
        if assessment.accuracy_score < 0.8:
            issues.append("Data Accuracy")
        
        if assessment.consistency_score < 0.8:
            issues.append("Data Consistency")
        
        if assessment.validity_score < 0.8:
            issues.append("Data Validity")
        
        return issues[:3]  # Return top 3 issues
    
    def _calculate_improvement_potential(
        self,
        assessment: QualityAssessment,
        unit: BusinessUnit
    ) -> float:
        """Calculate improvement potential score."""
        
        # Gap to perfect quality
        quality_gap = 1.0 - assessment.overall_score
        
        # Adjust based on unit characteristics
        if unit.quality_budget and unit.quality_budget > 0:
            investment_factor = min(2.0, unit.quality_budget / 50000)
        else:
            investment_factor = 0.5
        
        # Larger units have more resources for improvement
        size_factor = 1.0
        if unit.employee_count:
            if unit.employee_count > 500:
                size_factor = 1.3
            elif unit.employee_count > 100:
                size_factor = 1.1
        
        potential = quality_gap * investment_factor * size_factor * 100
        
        return min(100.0, potential)
    
    def _calculate_benchmark_position(self, assessment: QualityAssessment) -> str:
        """Calculate benchmark position for business unit."""
        
        score = assessment.overall_score
        
        if score >= 0.9:
            return "Industry Leader"
        elif score >= 0.8:
            return "Above Average"
        elif score >= 0.7:
            return "Average"
        elif score >= 0.6:
            return "Below Average"
        else:
            return "Needs Improvement"