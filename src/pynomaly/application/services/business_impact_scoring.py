"""Business impact scoring system for anomaly detection with financial analysis and ROI calculation."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field

from pynomaly.domain.entities import Dataset, Detector

logger = logging.getLogger(__name__)


class BusinessDomain(str, Enum):
    """Business domains for impact assessment."""

    REVENUE = "revenue"
    OPERATIONS = "operations"
    CUSTOMER_EXPERIENCE = "customer_experience"
    COMPLIANCE = "compliance"
    SECURITY = "security"
    REPUTATION = "reputation"
    SUPPLY_CHAIN = "supply_chain"
    INFRASTRUCTURE = "infrastructure"
    DATA_QUALITY = "data_quality"
    FRAUD_PREVENTION = "fraud_prevention"


class ImpactSeverity(str, Enum):
    """Impact severity levels."""

    NEGLIGIBLE = "negligible"  # < $1K impact
    LOW = "low"  # $1K - $10K impact
    MEDIUM = "medium"  # $10K - $100K impact
    HIGH = "high"  # $100K - $1M impact
    CRITICAL = "critical"  # > $1M impact


class RiskCategory(str, Enum):
    """Risk categories for business impact."""

    IMMEDIATE_LOSS = "immediate_loss"
    OPPORTUNITY_COST = "opportunity_cost"
    REMEDIATION_COST = "remediation_cost"
    REGULATORY_PENALTY = "regulatory_penalty"
    REPUTATION_DAMAGE = "reputation_damage"
    CUSTOMER_CHURN = "customer_churn"
    OPERATIONAL_DISRUPTION = "operational_disruption"
    COMPETITIVE_DISADVANTAGE = "competitive_disadvantage"


class TimeHorizon(str, Enum):
    """Time horizons for impact assessment."""

    IMMEDIATE = "immediate"  # 0-24 hours
    SHORT_TERM = "short_term"  # 1-7 days
    MEDIUM_TERM = "medium_term"  # 1-4 weeks
    LONG_TERM = "long_term"  # 1-12 months
    STRATEGIC = "strategic"  # > 12 months


@dataclass
class BusinessContext:
    """Business context for impact scoring."""

    industry: str
    revenue_annual: float = 0.0
    customer_base_size: int = 0
    transaction_volume_daily: int = 0
    average_transaction_value: float = 0.0
    compliance_requirements: List[str] = field(default_factory=list)
    critical_business_hours: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    peak_seasons: List[str] = field(default_factory=list)
    competitive_landscape: str = "moderate"  # "low", "moderate", "high"
    brand_value_score: float = 0.5  # 0.0 to 1.0


@dataclass
class ImpactMetric:
    """Individual impact metric calculation."""

    metric_name: str
    domain: BusinessDomain
    category: RiskCategory
    severity: ImpactSeverity
    financial_impact: float
    probability: float  # 0.0 to 1.0
    time_horizon: TimeHorizon
    confidence: float = 0.0  # 0.0 to 1.0
    evidence: List[str] = field(default_factory=list)
    mitigation_cost: float = 0.0
    recovery_time_hours: float = 0.0


@dataclass
class BusinessImpactScore:
    """Complete business impact assessment."""

    anomaly_id: str
    total_score: float  # 0.0 to 100.0
    financial_impact: float
    risk_level: ImpactSeverity
    confidence: float
    timestamp: datetime
    metrics: List[ImpactMetric] = field(default_factory=list)
    affected_domains: List[BusinessDomain] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    prevention_value: float = 0.0
    roi_detection: float = 0.0


@dataclass
class ROIAnalysis:
    """ROI analysis for anomaly detection system."""

    detection_system_cost: float
    prevented_losses: float
    false_positive_cost: float
    investigation_costs: float
    total_benefit: float
    total_cost: float
    roi_percentage: float
    payback_period_months: float
    cost_per_detection: float
    value_per_prevented_incident: float


class BusinessImpactModel:
    """Model for calculating business impact of anomalies."""

    def __init__(self, business_context: BusinessContext):
        self.business_context = business_context
        self.impact_history: List[BusinessImpactScore] = []
        self.domain_weights = self._initialize_domain_weights()
        self.severity_multipliers = self._initialize_severity_multipliers()

    def _initialize_domain_weights(self) -> Dict[BusinessDomain, float]:
        """Initialize domain importance weights based on business context."""
        base_weights = {
            BusinessDomain.REVENUE: 0.25,
            BusinessDomain.OPERATIONS: 0.20,
            BusinessDomain.CUSTOMER_EXPERIENCE: 0.15,
            BusinessDomain.COMPLIANCE: 0.10,
            BusinessDomain.SECURITY: 0.10,
            BusinessDomain.REPUTATION: 0.08,
            BusinessDomain.SUPPLY_CHAIN: 0.05,
            BusinessDomain.INFRASTRUCTURE: 0.03,
            BusinessDomain.DATA_QUALITY: 0.02,
            BusinessDomain.FRAUD_PREVENTION: 0.02,
        }

        # Adjust weights based on industry
        if "financial" in self.business_context.industry.lower():
            base_weights[BusinessDomain.COMPLIANCE] *= 2.0
            base_weights[BusinessDomain.FRAUD_PREVENTION] *= 3.0
            base_weights[BusinessDomain.SECURITY] *= 1.5
        elif "healthcare" in self.business_context.industry.lower():
            base_weights[BusinessDomain.COMPLIANCE] *= 2.5
            base_weights[BusinessDomain.DATA_QUALITY] *= 2.0
        elif "retail" in self.business_context.industry.lower():
            base_weights[BusinessDomain.CUSTOMER_EXPERIENCE] *= 1.5
            base_weights[BusinessDomain.REPUTATION] *= 1.3
        elif "manufacturing" in self.business_context.industry.lower():
            base_weights[BusinessDomain.SUPPLY_CHAIN] *= 2.0
            base_weights[BusinessDomain.OPERATIONS] *= 1.3

        # Normalize weights
        total_weight = sum(base_weights.values())
        return {
            domain: weight / total_weight for domain, weight in base_weights.items()
        }

    def _initialize_severity_multipliers(self) -> Dict[ImpactSeverity, float]:
        """Initialize severity multipliers for impact calculation."""
        return {
            ImpactSeverity.NEGLIGIBLE: 0.1,
            ImpactSeverity.LOW: 0.3,
            ImpactSeverity.MEDIUM: 1.0,
            ImpactSeverity.HIGH: 3.0,
            ImpactSeverity.CRITICAL: 10.0,
        }

    def calculate_revenue_impact(
        self,
        anomaly_data: Dict[str, Any],
        affected_transactions: int = 0,
        duration_hours: float = 1.0,
    ) -> ImpactMetric:
        """Calculate revenue impact from anomaly."""
        # Calculate direct revenue loss
        if affected_transactions > 0:
            direct_loss = (
                affected_transactions * self.business_context.average_transaction_value
            )
        else:
            # Estimate based on daily volume and duration
            hourly_transactions = self.business_context.transaction_volume_daily / 24
            affected_transactions = int(
                hourly_transactions * duration_hours * 0.1
            )  # 10% impact assumption
            direct_loss = (
                affected_transactions * self.business_context.average_transaction_value
            )

        # Add opportunity cost
        opportunity_multiplier = 1.2  # 20% opportunity cost
        total_impact = direct_loss * opportunity_multiplier

        # Determine severity
        severity = self._classify_financial_severity(total_impact)

        # Calculate probability based on anomaly confidence
        anomaly_confidence = anomaly_data.get("confidence", 0.5)
        probability = min(0.9, anomaly_confidence * 0.8)

        return ImpactMetric(
            metric_name="revenue_loss",
            domain=BusinessDomain.REVENUE,
            category=RiskCategory.IMMEDIATE_LOSS,
            severity=severity,
            financial_impact=total_impact,
            probability=probability,
            time_horizon=TimeHorizon.IMMEDIATE,
            confidence=anomaly_confidence,
            evidence=[
                f"Estimated {affected_transactions} affected transactions",
                f"Average transaction value: ${self.business_context.average_transaction_value:.2f}",
                f"Duration: {duration_hours:.1f} hours",
            ],
            recovery_time_hours=duration_hours,
        )

    def calculate_operational_impact(
        self,
        anomaly_data: Dict[str, Any],
        affected_systems: List[str],
        disruption_level: float = 0.5,
    ) -> ImpactMetric:
        """Calculate operational impact from anomaly."""
        # Base operational cost per hour (percentage of daily revenue)
        daily_revenue = self.business_context.revenue_annual / 365
        base_operational_cost = (
            daily_revenue * 0.1 / 24
        )  # 10% of daily revenue per day, divided by 24 hours

        # Calculate impact based on affected systems and disruption level
        system_multiplier = (
            len(affected_systems) * 0.3 + 0.7
        )  # More systems = higher impact
        total_impact = base_operational_cost * disruption_level * system_multiplier

        severity = self._classify_financial_severity(total_impact)
        probability = min(0.8, disruption_level)

        return ImpactMetric(
            metric_name="operational_disruption",
            domain=BusinessDomain.OPERATIONS,
            category=RiskCategory.OPERATIONAL_DISRUPTION,
            severity=severity,
            financial_impact=total_impact,
            probability=probability,
            time_horizon=TimeHorizon.SHORT_TERM,
            confidence=anomaly_data.get("confidence", 0.5),
            evidence=[
                f"Affected systems: {', '.join(affected_systems)}",
                f"Disruption level: {disruption_level * 100:.0f}%",
                f"System multiplier: {system_multiplier:.2f}",
            ],
            mitigation_cost=total_impact * 0.2,  # 20% of impact for mitigation
            recovery_time_hours=4.0,
        )

    def calculate_customer_impact(
        self,
        anomaly_data: Dict[str, Any],
        affected_customers: int = 0,
        service_degradation: float = 0.3,
    ) -> ImpactMetric:
        """Calculate customer experience impact."""
        if affected_customers == 0:
            # Estimate based on customer base
            affected_customers = int(
                self.business_context.customer_base_size * service_degradation * 0.1
            )

        # Calculate customer lifetime value impact
        annual_revenue_per_customer = self.business_context.revenue_annual / max(
            1, self.business_context.customer_base_size
        )
        customer_lifetime_value = (
            annual_revenue_per_customer * 3
        )  # Assume 3-year average lifetime

        # Calculate churn risk
        churn_risk = (
            service_degradation * 0.05
        )  # 5% base churn risk per degradation level
        churn_impact = affected_customers * churn_risk * customer_lifetime_value

        # Add satisfaction impact (harder to quantify, use percentage of CLV)
        satisfaction_impact = (
            affected_customers * customer_lifetime_value * 0.1 * service_degradation
        )

        total_impact = churn_impact + satisfaction_impact
        severity = self._classify_financial_severity(total_impact)

        return ImpactMetric(
            metric_name="customer_experience_degradation",
            domain=BusinessDomain.CUSTOMER_EXPERIENCE,
            category=RiskCategory.CUSTOMER_CHURN,
            severity=severity,
            financial_impact=total_impact,
            probability=service_degradation,
            time_horizon=TimeHorizon.MEDIUM_TERM,
            confidence=anomaly_data.get("confidence", 0.5),
            evidence=[
                f"Affected customers: {affected_customers:,}",
                f"Service degradation: {service_degradation * 100:.0f}%",
                f"Estimated churn risk: {churn_risk * 100:.1f}%",
                f"Customer lifetime value: ${customer_lifetime_value:.2f}",
            ],
            recovery_time_hours=24.0,
        )

    def calculate_compliance_impact(
        self,
        anomaly_data: Dict[str, Any],
        violated_requirements: List[str],
        data_records_affected: int = 0,
    ) -> ImpactMetric:
        """Calculate compliance and regulatory impact."""
        base_penalty = 0.0

        # Calculate penalties based on violated requirements
        for requirement in violated_requirements:
            if "gdpr" in requirement.lower():
                base_penalty += max(
                    10000, data_records_affected * 2
                )  # €2 per record, min €10K
            elif "hipaa" in requirement.lower():
                base_penalty += data_records_affected * 50  # $50 per record
            elif "pci" in requirement.lower():
                base_penalty += 50000  # Base $50K for PCI violations
            elif "sox" in requirement.lower():
                base_penalty += 100000  # Base $100K for SOX violations
            else:
                base_penalty += 25000  # Generic regulatory penalty

        # Add remediation costs
        remediation_cost = base_penalty * 0.5  # 50% of penalty for remediation
        total_impact = base_penalty + remediation_cost

        severity = self._classify_financial_severity(total_impact)
        probability = 0.7 if violated_requirements else 0.1

        return ImpactMetric(
            metric_name="compliance_violation",
            domain=BusinessDomain.COMPLIANCE,
            category=RiskCategory.REGULATORY_PENALTY,
            severity=severity,
            financial_impact=total_impact,
            probability=probability,
            time_horizon=TimeHorizon.LONG_TERM,
            confidence=anomaly_data.get("confidence", 0.5),
            evidence=[
                f"Violated requirements: {', '.join(violated_requirements)}",
                f"Affected data records: {data_records_affected:,}",
                f"Base penalty: ${base_penalty:.2f}",
                f"Remediation cost: ${remediation_cost:.2f}",
            ],
            mitigation_cost=remediation_cost,
            recovery_time_hours=168.0,  # 1 week
        )

    def calculate_security_impact(
        self,
        anomaly_data: Dict[str, Any],
        breach_probability: float = 0.1,
        data_sensitivity: str = "medium",
    ) -> ImpactMetric:
        """Calculate security-related business impact."""
        # Base security incident costs
        base_costs = {
            "low": 50000,  # $50K for low sensitivity data
            "medium": 200000,  # $200K for medium sensitivity
            "high": 1000000,  # $1M for high sensitivity
            "critical": 5000000,  # $5M for critical data
        }

        base_cost = base_costs.get(data_sensitivity, 200000)

        # Scale by company size
        company_size_multiplier = min(
            3.0, self.business_context.revenue_annual / 100_000_000
        )  # Scale with revenue

        # Add investigation and remediation costs
        investigation_cost = base_cost * 0.3
        remediation_cost = base_cost * 0.4
        legal_cost = base_cost * 0.2

        total_impact = (
            base_cost + investigation_cost + remediation_cost + legal_cost
        ) * company_size_multiplier

        severity = self._classify_financial_severity(total_impact)

        return ImpactMetric(
            metric_name="security_breach_risk",
            domain=BusinessDomain.SECURITY,
            category=RiskCategory.IMMEDIATE_LOSS,
            severity=severity,
            financial_impact=total_impact,
            probability=breach_probability,
            time_horizon=TimeHorizon.IMMEDIATE,
            confidence=anomaly_data.get("confidence", 0.5),
            evidence=[
                f"Data sensitivity: {data_sensitivity}",
                f"Breach probability: {breach_probability * 100:.1f}%",
                f"Company size multiplier: {company_size_multiplier:.2f}",
                f"Base incident cost: ${base_cost:,.2f}",
            ],
            mitigation_cost=base_cost * 0.1,  # 10% for prevention
            recovery_time_hours=72.0,  # 3 days
        )

    def _classify_financial_severity(self, financial_impact: float) -> ImpactSeverity:
        """Classify financial impact into severity levels."""
        if financial_impact < 1000:
            return ImpactSeverity.NEGLIGIBLE
        elif financial_impact < 10000:
            return ImpactSeverity.LOW
        elif financial_impact < 100000:
            return ImpactSeverity.MEDIUM
        elif financial_impact < 1000000:
            return ImpactSeverity.HIGH
        else:
            return ImpactSeverity.CRITICAL


class BusinessImpactAnalyzer:
    """Main service for business impact analysis and scoring."""

    def __init__(self, business_context: BusinessContext):
        self.business_context = business_context
        self.impact_model = BusinessImpactModel(business_context)
        self.scoring_history: Dict[str, List[BusinessImpactScore]] = defaultdict(list)

    async def analyze_anomaly_impact(
        self,
        anomaly_id: str,
        anomaly_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> BusinessImpactScore:
        """Analyze business impact of an anomaly."""
        context = context or {}
        metrics = []

        # Calculate impact across different domains
        try:
            # Revenue impact
            revenue_metric = self.impact_model.calculate_revenue_impact(
                anomaly_data,
                affected_transactions=context.get("affected_transactions", 0),
                duration_hours=context.get("duration_hours", 1.0),
            )
            metrics.append(revenue_metric)

            # Operational impact
            if context.get("affected_systems"):
                operational_metric = self.impact_model.calculate_operational_impact(
                    anomaly_data,
                    affected_systems=context.get("affected_systems", []),
                    disruption_level=context.get("disruption_level", 0.5),
                )
                metrics.append(operational_metric)

            # Customer impact
            if context.get("affects_customers", True):
                customer_metric = self.impact_model.calculate_customer_impact(
                    anomaly_data,
                    affected_customers=context.get("affected_customers", 0),
                    service_degradation=context.get("service_degradation", 0.3),
                )
                metrics.append(customer_metric)

            # Compliance impact
            if context.get("compliance_violations"):
                compliance_metric = self.impact_model.calculate_compliance_impact(
                    anomaly_data,
                    violated_requirements=context.get("compliance_violations", []),
                    data_records_affected=context.get("data_records_affected", 0),
                )
                metrics.append(compliance_metric)

            # Security impact
            if context.get("security_relevant", False):
                security_metric = self.impact_model.calculate_security_impact(
                    anomaly_data,
                    breach_probability=context.get("breach_probability", 0.1),
                    data_sensitivity=context.get("data_sensitivity", "medium"),
                )
                metrics.append(security_metric)

        except Exception as e:
            logger.error(f"Error calculating impact metrics: {e}")
            # Provide fallback metric
            metrics.append(
                ImpactMetric(
                    metric_name="fallback_assessment",
                    domain=BusinessDomain.OPERATIONS,
                    category=RiskCategory.OPERATIONAL_DISRUPTION,
                    severity=ImpactSeverity.MEDIUM,
                    financial_impact=50000.0,
                    probability=0.5,
                    time_horizon=TimeHorizon.SHORT_TERM,
                    confidence=0.3,
                )
            )

        # Calculate overall score
        total_score = self._calculate_composite_score(metrics)
        total_financial_impact = sum(
            m.financial_impact * m.probability for m in metrics
        )
        risk_level = self._determine_overall_risk_level(metrics)
        overall_confidence = (
            np.mean([m.confidence for m in metrics]) if metrics else 0.0
        )

        # Determine affected domains
        affected_domains = list(set(m.domain for m in metrics))

        # Generate recommendations
        recommendations = self._generate_recommendations(
            metrics, total_financial_impact
        )

        # Calculate prevention value (what we save by detecting this)
        prevention_value = total_financial_impact * 0.8  # Assume 80% preventable

        impact_score = BusinessImpactScore(
            anomaly_id=anomaly_id,
            total_score=total_score,
            financial_impact=total_financial_impact,
            risk_level=risk_level,
            confidence=overall_confidence,
            timestamp=datetime.now(),
            metrics=metrics,
            affected_domains=affected_domains,
            recommended_actions=recommendations,
            prevention_value=prevention_value,
        )

        # Store in history
        self.scoring_history[anomaly_id].append(impact_score)
        self.impact_model.impact_history.append(impact_score)

        return impact_score

    def _calculate_composite_score(self, metrics: List[ImpactMetric]) -> float:
        """Calculate composite business impact score (0-100)."""
        if not metrics:
            return 0.0

        weighted_score = 0.0
        total_weight = 0.0

        for metric in metrics:
            # Get domain weight
            domain_weight = self.impact_model.domain_weights.get(metric.domain, 0.1)

            # Get severity multiplier
            severity_multiplier = self.impact_model.severity_multipliers.get(
                metric.severity, 1.0
            )

            # Calculate metric score (0-100)
            metric_score = (
                (metric.probability * 0.4)  # 40% probability weight
                + (metric.confidence * 0.3)  # 30% confidence weight
                + (severity_multiplier / 10.0 * 0.3)  # 30% severity weight
            ) * 100

            weighted_score += metric_score * domain_weight
            total_weight += domain_weight

        return min(100.0, weighted_score / total_weight if total_weight > 0 else 0.0)

    def _determine_overall_risk_level(
        self, metrics: List[ImpactMetric]
    ) -> ImpactSeverity:
        """Determine overall risk level from metrics."""
        if not metrics:
            return ImpactSeverity.LOW

        # Get highest severity with significant probability
        significant_metrics = [m for m in metrics if m.probability > 0.3]
        if not significant_metrics:
            significant_metrics = metrics

        severity_values = {
            ImpactSeverity.NEGLIGIBLE: 1,
            ImpactSeverity.LOW: 2,
            ImpactSeverity.MEDIUM: 3,
            ImpactSeverity.HIGH: 4,
            ImpactSeverity.CRITICAL: 5,
        }

        max_severity_value = max(
            severity_values[m.severity] for m in significant_metrics
        )

        for severity, value in severity_values.items():
            if value == max_severity_value:
                return severity

        return ImpactSeverity.MEDIUM

    def _generate_recommendations(
        self, metrics: List[ImpactMetric], total_impact: float
    ) -> List[str]:
        """Generate action recommendations based on impact analysis."""
        recommendations = []

        # Priority-based recommendations
        if total_impact > 1000000:  # > $1M impact
            recommendations.extend(
                [
                    "Escalate to executive leadership immediately",
                    "Activate incident response team",
                    "Consider emergency system shutdown if necessary",
                ]
            )
        elif total_impact > 100000:  # > $100K impact
            recommendations.extend(
                [
                    "Escalate to senior management",
                    "Initiate formal incident response",
                    "Notify key stakeholders",
                ]
            )
        else:
            recommendations.extend(
                [
                    "Investigate anomaly promptly",
                    "Monitor for escalation",
                    "Document findings",
                ]
            )

        # Domain-specific recommendations
        affected_domains = set(m.domain for m in metrics)

        if BusinessDomain.REVENUE in affected_domains:
            recommendations.append("Review revenue impact and recovery procedures")

        if BusinessDomain.COMPLIANCE in affected_domains:
            recommendations.extend(
                [
                    "Notify compliance team",
                    "Prepare regulatory reports if required",
                    "Review legal implications",
                ]
            )

        if BusinessDomain.SECURITY in affected_domains:
            recommendations.extend(
                [
                    "Engage security team",
                    "Check for signs of breach",
                    "Review access logs",
                ]
            )

        if BusinessDomain.CUSTOMER_EXPERIENCE in affected_domains:
            recommendations.extend(
                [
                    "Prepare customer communication",
                    "Monitor customer support channels",
                    "Consider service credits if applicable",
                ]
            )

        return list(set(recommendations))  # Remove duplicates

    async def calculate_detection_roi(
        self, time_period_days: int = 365, detection_system_cost: float = 0.0
    ) -> ROIAnalysis:
        """Calculate ROI for the anomaly detection system."""
        # Get impacts from the specified time period
        cutoff_date = datetime.now() - timedelta(days=time_period_days)
        recent_impacts = [
            impact
            for impact in self.impact_model.impact_history
            if impact.timestamp >= cutoff_date
        ]

        if not recent_impacts:
            return ROIAnalysis(
                detection_system_cost=detection_system_cost,
                prevented_losses=0.0,
                false_positive_cost=0.0,
                investigation_costs=0.0,
                total_benefit=0.0,
                total_cost=detection_system_cost,
                roi_percentage=0.0,
                payback_period_months=float("inf"),
                cost_per_detection=detection_system_cost,
                value_per_prevented_incident=0.0,
            )

        # Calculate prevented losses
        prevented_losses = sum(impact.prevention_value for impact in recent_impacts)

        # Estimate investigation costs (assume $1000 per anomaly investigation)
        investigation_costs = len(recent_impacts) * 1000

        # Estimate false positive costs (assume 20% false positive rate, $500 per false positive)
        estimated_false_positives = len(recent_impacts) * 0.2
        false_positive_cost = estimated_false_positives * 500

        # Calculate total costs and benefits
        total_cost = detection_system_cost + investigation_costs + false_positive_cost
        total_benefit = prevented_losses

        # Calculate ROI metrics
        net_benefit = total_benefit - total_cost
        roi_percentage = (net_benefit / total_cost * 100) if total_cost > 0 else 0.0

        # Calculate payback period
        monthly_benefit = (
            total_benefit / 12
            if time_period_days >= 365
            else total_benefit * (30 / time_period_days)
        )
        monthly_cost = (
            total_cost / 12
            if time_period_days >= 365
            else total_cost * (30 / time_period_days)
        )
        payback_period_months = (
            (detection_system_cost / monthly_benefit)
            if monthly_benefit > 0
            else float("inf")
        )

        # Calculate per-unit metrics
        cost_per_detection = total_cost / len(recent_impacts) if recent_impacts else 0.0
        value_per_prevented_incident = (
            prevented_losses / len(recent_impacts) if recent_impacts else 0.0
        )

        return ROIAnalysis(
            detection_system_cost=detection_system_cost,
            prevented_losses=prevented_losses,
            false_positive_cost=false_positive_cost,
            investigation_costs=investigation_costs,
            total_benefit=total_benefit,
            total_cost=total_cost,
            roi_percentage=roi_percentage,
            payback_period_months=payback_period_months,
            cost_per_detection=cost_per_detection,
            value_per_prevented_incident=value_per_prevented_incident,
        )

    async def get_impact_trends(self, time_period_days: int = 30) -> Dict[str, Any]:
        """Get impact trends and analytics."""
        cutoff_date = datetime.now() - timedelta(days=time_period_days)
        recent_impacts = [
            impact
            for impact in self.impact_model.impact_history
            if impact.timestamp >= cutoff_date
        ]

        if not recent_impacts:
            return {
                "total_incidents": 0,
                "total_financial_impact": 0.0,
                "average_impact_per_incident": 0.0,
                "domain_breakdown": {},
                "severity_breakdown": {},
                "trends": {},
            }

        # Calculate basic metrics
        total_incidents = len(recent_impacts)
        total_financial_impact = sum(
            impact.financial_impact for impact in recent_impacts
        )
        average_impact = total_financial_impact / total_incidents

        # Domain breakdown
        domain_impacts = defaultdict(float)
        for impact in recent_impacts:
            for domain in impact.affected_domains:
                domain_impacts[domain.value] += impact.financial_impact

        # Severity breakdown
        severity_counts = defaultdict(int)
        for impact in recent_impacts:
            severity_counts[impact.risk_level.value] += 1

        # Trend analysis (weekly breakdown)
        weekly_trends = defaultdict(lambda: {"count": 0, "impact": 0.0})
        for impact in recent_impacts:
            week_key = impact.timestamp.strftime("%Y-W%U")
            weekly_trends[week_key]["count"] += 1
            weekly_trends[week_key]["impact"] += impact.financial_impact

        return {
            "total_incidents": total_incidents,
            "total_financial_impact": total_financial_impact,
            "average_impact_per_incident": average_impact,
            "domain_breakdown": dict(domain_impacts),
            "severity_breakdown": dict(severity_counts),
            "weekly_trends": dict(weekly_trends),
            "prevention_value": sum(
                impact.prevention_value for impact in recent_impacts
            ),
        }
