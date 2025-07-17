"""Real-time decision support system for automated anomaly response and recommendation engine."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np

from .business_impact_scoring import (
    BusinessImpactAnalyzer,
    BusinessImpactScore,
    ImpactSeverity,
)

logger = logging.getLogger(__name__)


class DecisionType(str, Enum):
    """Types of automated decisions."""

    ESCALATE = "escalate"
    INVESTIGATE = "investigate"
    MITIGATE = "mitigate"
    IGNORE = "ignore"
    QUARANTINE = "quarantine"
    ALERT = "alert"
    AUTO_REMEDIATE = "auto_remediate"
    MANUAL_REVIEW = "manual_review"


class ConfidenceLevel(str, Enum):
    """Confidence levels for decisions."""

    VERY_LOW = "very_low"  # < 30%
    LOW = "low"  # 30-50%
    MEDIUM = "medium"  # 50-70%
    HIGH = "high"  # 70-90%
    VERY_HIGH = "very_high"  # > 90%


class AutomationLevel(str, Enum):
    """Levels of automation for responses."""

    MANUAL_ONLY = "manual_only"
    RECOMMEND_ONLY = "recommend_only"
    SEMI_AUTOMATED = "semi_automated"
    FULLY_AUTOMATED = "fully_automated"
    EMERGENCY_ONLY = "emergency_only"


class ResponseStatus(str, Enum):
    """Status of response execution."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class DecisionContext:
    """Context information for decision making."""

    anomaly_id: str
    timestamp: datetime
    anomaly_data: dict[str, Any]
    business_impact: BusinessImpactScore | None = None
    historical_patterns: dict[str, Any] = field(default_factory=dict)
    system_state: dict[str, Any] = field(default_factory=dict)
    user_preferences: dict[str, Any] = field(default_factory=dict)
    organizational_policies: dict[str, Any] = field(default_factory=dict)
    resource_availability: dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionRule:
    """Rule for automated decision making."""

    rule_id: str
    name: str
    conditions: list[dict[str, Any]]
    decision: DecisionType
    confidence_threshold: float = 0.7
    automation_level: AutomationLevel = AutomationLevel.RECOMMEND_ONLY
    priority: int = 1  # 1-10, higher is more important
    enabled: bool = True
    cooldown_minutes: int = 0  # Minimum time between rule applications
    max_executions_per_hour: int = 100
    execution_count: int = 0
    last_executed: datetime | None = None


@dataclass
class ResponseAction:
    """Automated response action."""

    action_id: str
    action_type: str
    description: str
    target_systems: list[str]
    parameters: dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 300
    retry_count: int = 3
    requires_approval: bool = False
    rollback_action: str | None = None


@dataclass
class DecisionRecommendation:
    """Decision recommendation with supporting evidence."""

    recommendation_id: str
    decision: DecisionType
    confidence: ConfidenceLevel
    reasoning: list[str]
    evidence: dict[str, Any]
    recommended_actions: list[ResponseAction]
    estimated_impact: float
    time_sensitivity: str  # "immediate", "urgent", "normal", "low"
    resource_requirements: dict[str, Any] = field(default_factory=dict)
    alternatives: list[dict[str, Any]] = field(default_factory=list)
    risk_assessment: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """Result of executing a decision or action."""

    execution_id: str
    status: ResponseStatus
    start_time: datetime
    end_time: datetime | None = None
    success: bool = False
    error_message: str | None = None
    output_data: dict[str, Any] = field(default_factory=dict)
    measurements: dict[str, float] = field(default_factory=dict)


class DecisionEngine:
    """Core decision engine for evaluating anomalies and generating recommendations."""

    def __init__(self):
        self.rules: dict[str, DecisionRule] = {}
        self.response_actions: dict[str, ResponseAction] = {}
        self.execution_history: list[ExecutionResult] = []
        self.rule_performance: dict[str, dict[str, float]] = defaultdict(dict)
        self._initialize_default_rules()
        self._initialize_default_actions()

    def _initialize_default_rules(self) -> None:
        """Initialize default decision rules."""
        # High impact, high confidence rule
        self.rules["high_impact_escalate"] = DecisionRule(
            rule_id="high_impact_escalate",
            name="Escalate High Impact Anomalies",
            conditions=[
                {
                    "field": "business_impact.risk_level",
                    "operator": "in",
                    "value": ["high", "critical"],
                },
                {"field": "anomaly_data.confidence", "operator": ">=", "value": 0.7},
            ],
            decision=DecisionType.ESCALATE,
            confidence_threshold=0.8,
            automation_level=AutomationLevel.SEMI_AUTOMATED,
            priority=9,
        )

        # Security-related immediate action
        self.rules["security_quarantine"] = DecisionRule(
            rule_id="security_quarantine",
            name="Quarantine Security Threats",
            conditions=[
                {
                    "field": "anomaly_data.category",
                    "operator": "==",
                    "value": "security",
                },
                {"field": "anomaly_data.confidence", "operator": ">=", "value": 0.8},
                {
                    "field": "business_impact.affected_domains",
                    "operator": "contains",
                    "value": "security",
                },
            ],
            decision=DecisionType.QUARANTINE,
            confidence_threshold=0.9,
            automation_level=AutomationLevel.FULLY_AUTOMATED,
            priority=10,
            cooldown_minutes=5,
        )

        # Low impact monitoring
        self.rules["low_impact_monitor"] = DecisionRule(
            rule_id="low_impact_monitor",
            name="Monitor Low Impact Anomalies",
            conditions=[
                {
                    "field": "business_impact.risk_level",
                    "operator": "in",
                    "value": ["negligible", "low"],
                },
                {"field": "anomaly_data.confidence", "operator": "<", "value": 0.6},
            ],
            decision=DecisionType.INVESTIGATE,
            confidence_threshold=0.5,
            automation_level=AutomationLevel.FULLY_AUTOMATED,
            priority=3,
        )

        # Compliance violations
        self.rules["compliance_immediate"] = DecisionRule(
            rule_id="compliance_immediate",
            name="Immediate Action for Compliance Violations",
            conditions=[
                {
                    "field": "business_impact.affected_domains",
                    "operator": "contains",
                    "value": "compliance",
                },
                {"field": "anomaly_data.confidence", "operator": ">=", "value": 0.6},
            ],
            decision=DecisionType.ESCALATE,
            confidence_threshold=0.7,
            automation_level=AutomationLevel.SEMI_AUTOMATED,
            priority=8,
        )

        # Revenue impact mitigation
        self.rules["revenue_mitigation"] = DecisionRule(
            rule_id="revenue_mitigation",
            name="Mitigate Revenue Impact",
            conditions=[
                {
                    "field": "business_impact.affected_domains",
                    "operator": "contains",
                    "value": "revenue",
                },
                {
                    "field": "business_impact.financial_impact",
                    "operator": ">=",
                    "value": 50000,
                },
            ],
            decision=DecisionType.MITIGATE,
            confidence_threshold=0.6,
            automation_level=AutomationLevel.SEMI_AUTOMATED,
            priority=7,
        )

    def _initialize_default_actions(self) -> None:
        """Initialize default response actions."""
        self.response_actions["notify_admin"] = ResponseAction(
            action_id="notify_admin",
            action_type="notification",
            description="Send notification to system administrators",
            target_systems=["notification_service"],
            parameters={"channel": "email", "priority": "high"},
            timeout_seconds=30,
        )

        self.response_actions["block_ip"] = ResponseAction(
            action_id="block_ip",
            action_type="network_security",
            description="Block suspicious IP address",
            target_systems=["firewall", "waf"],
            parameters={"action": "block", "duration_hours": 24},
            timeout_seconds=60,
            requires_approval=False,
            rollback_action="unblock_ip",
        )

        self.response_actions["isolate_system"] = ResponseAction(
            action_id="isolate_system",
            action_type="system_security",
            description="Isolate affected system from network",
            target_systems=["network_controller"],
            parameters={"isolation_level": "full"},
            timeout_seconds=120,
            requires_approval=True,
            rollback_action="restore_system_access",
        )

        self.response_actions["scale_resources"] = ResponseAction(
            action_id="scale_resources",
            action_type="capacity_management",
            description="Scale system resources to handle load",
            target_systems=["orchestrator", "load_balancer"],
            parameters={"scale_factor": 1.5, "max_instances": 10},
            timeout_seconds=300,
        )

        self.response_actions["backup_data"] = ResponseAction(
            action_id="backup_data",
            action_type="data_protection",
            description="Create emergency backup of critical data",
            target_systems=["backup_service"],
            parameters={"backup_type": "emergency", "retention_days": 30},
            timeout_seconds=600,
        )

    def evaluate_conditions(
        self, conditions: list[dict[str, Any]], context: DecisionContext
    ) -> bool:
        """Evaluate if conditions are met for a rule."""
        for condition in conditions:
            field_path = condition["field"]
            operator = condition["operator"]
            expected_value = condition["value"]

            # Get actual value from context
            actual_value = self._get_field_value(field_path, context)

            # Evaluate condition
            if not self._evaluate_condition(actual_value, operator, expected_value):
                return False

        return True

    def _get_field_value(self, field_path: str, context: DecisionContext) -> Any:
        """Extract field value from context using dot notation."""
        parts = field_path.split(".")
        value = context

        for part in parts:
            if hasattr(value, part):
                value = getattr(value, part)
            elif isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return None

        return value

    def _evaluate_condition(self, actual: Any, operator: str, expected: Any) -> bool:
        """Evaluate a single condition."""
        try:
            if operator == "==":
                return actual == expected
            elif operator == "!=":
                return actual != expected
            elif operator == "<":
                return actual < expected
            elif operator == "<=":
                return actual <= expected
            elif operator == ">":
                return actual > expected
            elif operator == ">=":
                return actual >= expected
            elif operator == "in":
                return actual in expected
            elif operator == "not_in":
                return actual not in expected
            elif operator == "contains":
                return expected in actual if isinstance(actual, (list, str)) else False
            elif operator == "starts_with":
                return str(actual).startswith(str(expected))
            elif operator == "ends_with":
                return str(actual).endswith(str(expected))
            else:
                logger.warning(f"Unknown operator: {operator}")
                return False
        except Exception as e:
            logger.error(f"Error evaluating condition: {e}")
            return False

    def generate_recommendation(
        self, context: DecisionContext
    ) -> DecisionRecommendation | None:
        """Generate decision recommendation based on context."""
        applicable_rules = []

        # Find applicable rules
        for rule in self.rules.values():
            if not rule.enabled:
                continue

            # Check cooldown
            if rule.last_executed and rule.cooldown_minutes > 0:
                time_since_last = (
                    datetime.now() - rule.last_executed
                ).total_seconds() / 60
                if time_since_last < rule.cooldown_minutes:
                    continue

            # Check rate limit
            if rule.max_executions_per_hour > 0:
                hour_ago = datetime.now() - timedelta(hours=1)
                recent_executions = sum(
                    1
                    for result in self.execution_history
                    if result.start_time > hour_ago
                    and rule.rule_id in result.output_data.get("rule_id", "")
                )
                if recent_executions >= rule.max_executions_per_hour:
                    continue

            # Evaluate conditions
            if self.evaluate_conditions(rule.conditions, context):
                applicable_rules.append(rule)

        if not applicable_rules:
            return self._generate_default_recommendation(context)

        # Sort by priority (highest first)
        applicable_rules.sort(key=lambda r: r.priority, reverse=True)

        # Select best rule
        best_rule = applicable_rules[0]

        # Calculate confidence
        confidence = self._calculate_decision_confidence(best_rule, context)
        confidence_level = self._get_confidence_level(confidence)

        # Generate reasoning
        reasoning = self._generate_reasoning(best_rule, context)

        # Get recommended actions
        recommended_actions = self._get_actions_for_decision(
            best_rule.decision, context
        )

        # Estimate impact
        estimated_impact = self._estimate_decision_impact(best_rule.decision, context)

        # Determine time sensitivity
        time_sensitivity = self._determine_time_sensitivity(context)

        # Generate alternatives
        alternatives = self._generate_alternatives(applicable_rules[1:], context)

        # Risk assessment
        risk_assessment = self._assess_decision_risk(best_rule.decision, context)

        return DecisionRecommendation(
            recommendation_id=f"rec_{context.anomaly_id}_{int(datetime.now().timestamp())}",
            decision=best_rule.decision,
            confidence=confidence_level,
            reasoning=reasoning,
            evidence={
                "rule_id": best_rule.rule_id,
                "rule_name": best_rule.name,
                "conditions_met": [
                    c
                    for c in best_rule.conditions
                    if self._evaluate_condition(
                        self._get_field_value(c["field"], context),
                        c["operator"],
                        c["value"],
                    )
                ],
                "business_impact": context.business_impact.__dict__
                if context.business_impact
                else None,
                "anomaly_confidence": context.anomaly_data.get("confidence", 0.0),
            },
            recommended_actions=recommended_actions,
            estimated_impact=estimated_impact,
            time_sensitivity=time_sensitivity,
            alternatives=alternatives,
            risk_assessment=risk_assessment,
        )

    def _generate_default_recommendation(
        self, context: DecisionContext
    ) -> DecisionRecommendation:
        """Generate default recommendation when no rules apply."""
        # Default to investigation for unknown cases
        return DecisionRecommendation(
            recommendation_id=f"default_rec_{context.anomaly_id}_{int(datetime.now().timestamp())}",
            decision=DecisionType.INVESTIGATE,
            confidence=ConfidenceLevel.LOW,
            reasoning=[
                "No specific rules matched this anomaly",
                "Defaulting to investigation",
            ],
            evidence={"default_case": True},
            recommended_actions=[self.response_actions["notify_admin"]],
            estimated_impact=0.0,
            time_sensitivity="normal",
        )

    def _calculate_decision_confidence(
        self, rule: DecisionRule, context: DecisionContext
    ) -> float:
        """Calculate confidence in the decision."""
        base_confidence = rule.confidence_threshold

        # Adjust based on anomaly confidence
        anomaly_confidence = context.anomaly_data.get("confidence", 0.5)
        confidence_adjustment = (anomaly_confidence - 0.5) * 0.2  # ±0.1 adjustment

        # Adjust based on business impact clarity
        if context.business_impact:
            impact_confidence = context.business_impact.confidence
            confidence_adjustment += (impact_confidence - 0.5) * 0.1

        # Adjust based on rule performance history
        rule_performance = self.rule_performance.get(rule.rule_id, {})
        success_rate = rule_performance.get("success_rate", 0.5)
        confidence_adjustment += (success_rate - 0.5) * 0.15

        return min(1.0, max(0.0, base_confidence + confidence_adjustment))

    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Convert numeric confidence to confidence level."""
        if confidence < 0.3:
            return ConfidenceLevel.VERY_LOW
        elif confidence < 0.5:
            return ConfidenceLevel.LOW
        elif confidence < 0.7:
            return ConfidenceLevel.MEDIUM
        elif confidence < 0.9:
            return ConfidenceLevel.HIGH
        else:
            return ConfidenceLevel.VERY_HIGH

    def _generate_reasoning(
        self, rule: DecisionRule, context: DecisionContext
    ) -> list[str]:
        """Generate human-readable reasoning for the decision."""
        reasoning = [f"Applied rule: {rule.name}"]

        # Add business impact reasoning
        if context.business_impact:
            reasoning.append(
                f"Business impact level: {context.business_impact.risk_level.value}"
            )
            if context.business_impact.financial_impact > 0:
                reasoning.append(
                    f"Estimated financial impact: ${context.business_impact.financial_impact:,.2f}"
                )

        # Add anomaly confidence reasoning
        anomaly_confidence = context.anomaly_data.get("confidence", 0.0)
        reasoning.append(f"Anomaly processing confidence: {anomaly_confidence:.1%}")

        # Add specific condition reasoning
        for condition in rule.conditions:
            field_value = self._get_field_value(condition["field"], context)
            reasoning.append(
                f"Condition met: {condition['field']} {condition['operator']} {condition['value']} (actual: {field_value})"
            )

        return reasoning

    def _get_actions_for_decision(
        self, decision: DecisionType, context: DecisionContext
    ) -> list[ResponseAction]:
        """Get appropriate response actions for a decision type."""
        action_mapping = {
            DecisionType.ESCALATE: ["notify_admin"],
            DecisionType.INVESTIGATE: ["notify_admin"],
            DecisionType.MITIGATE: ["scale_resources", "notify_admin"],
            DecisionType.QUARANTINE: ["isolate_system", "block_ip", "notify_admin"],
            DecisionType.AUTO_REMEDIATE: ["scale_resources", "backup_data"],
            DecisionType.ALERT: ["notify_admin"],
            DecisionType.IGNORE: [],
            DecisionType.MANUAL_REVIEW: ["notify_admin"],
        }

        action_ids = action_mapping.get(decision, ["notify_admin"])
        return [
            self.response_actions[action_id]
            for action_id in action_ids
            if action_id in self.response_actions
        ]

    def _estimate_decision_impact(
        self, decision: DecisionType, context: DecisionContext
    ) -> float:
        """Estimate the impact of executing the decision."""
        # Base impact estimates (can be negative for beneficial actions)
        impact_estimates = {
            DecisionType.ESCALATE: -1000,  # Cost of escalation
            DecisionType.INVESTIGATE: -500,  # Cost of investigation
            DecisionType.MITIGATE: -5000,  # Cost of mitigation
            DecisionType.QUARANTINE: -10000,  # Cost of quarantine
            DecisionType.AUTO_REMEDIATE: -2000,  # Cost of auto-remediation
            DecisionType.ALERT: -100,  # Cost of alerting
            DecisionType.IGNORE: 0,  # No immediate cost
            DecisionType.MANUAL_REVIEW: -1500,  # Cost of manual review
        }

        base_impact = impact_estimates.get(decision, 0)

        # Adjust based on business impact
        if context.business_impact:
            # If we prevent a high impact, the decision has positive value
            prevention_value = (
                context.business_impact.financial_impact * 0.8
            )  # 80% prevention assumption
            return base_impact + prevention_value

        return base_impact

    def _determine_time_sensitivity(self, context: DecisionContext) -> str:
        """Determine time sensitivity of the decision."""
        if context.business_impact:
            if context.business_impact.risk_level in [
                ImpactSeverity.HIGH,
                ImpactSeverity.CRITICAL,
            ]:
                return "immediate"
            elif context.business_impact.risk_level == ImpactSeverity.MEDIUM:
                return "urgent"

        anomaly_confidence = context.anomaly_data.get("confidence", 0.0)
        if anomaly_confidence > 0.8:
            return "urgent"
        elif anomaly_confidence > 0.6:
            return "normal"
        else:
            return "low"

    def _generate_alternatives(
        self, other_rules: list[DecisionRule], context: DecisionContext
    ) -> list[dict[str, Any]]:
        """Generate alternative decision options."""
        alternatives = []

        for rule in other_rules[:3]:  # Top 3 alternatives
            confidence = self._calculate_decision_confidence(rule, context)
            alternatives.append(
                {
                    "decision": rule.decision.value,
                    "rule_name": rule.name,
                    "confidence": confidence,
                    "reasoning": f"Alternative based on {rule.name}",
                }
            )

        return alternatives

    def _assess_decision_risk(
        self, decision: DecisionType, context: DecisionContext
    ) -> dict[str, Any]:
        """Assess risks of the proposed decision."""
        risk_factors = {
            DecisionType.QUARANTINE: {
                "service_disruption": "high",
                "false_positive_impact": "high",
                "recovery_complexity": "medium",
            },
            DecisionType.AUTO_REMEDIATE: {
                "automation_failure": "medium",
                "unintended_consequences": "medium",
                "resource_usage": "high",
            },
            DecisionType.ESCALATE: {
                "response_delay": "low",
                "resource_consumption": "medium",
                "alert_fatigue": "low",
            },
            DecisionType.IGNORE: {
                "missed_threat": "high",
                "business_impact": "high",
                "compliance_risk": "medium",
            },
        }

        base_risks = risk_factors.get(decision, {"general_risk": "low"})

        # Add context-specific risks
        if (
            context.business_impact
            and context.business_impact.risk_level == ImpactSeverity.CRITICAL
        ):
            base_risks["business_continuity"] = "critical"

        return base_risks


class RealTimeDecisionSupport:
    """Main service for real-time decision support and automated response."""

    def __init__(self, business_impact_analyzer: BusinessImpactAnalyzer):
        self.business_impact_analyzer = business_impact_analyzer
        self.decision_engine = DecisionEngine()
        self.active_decisions: dict[str, DecisionRecommendation] = {}
        self.execution_queue: deque = deque()
        self.automation_policies: dict[str, Any] = self._load_automation_policies()

    def _load_automation_policies(self) -> dict[str, Any]:
        """Load automation policies and constraints."""
        return {
            "max_concurrent_auto_actions": 5,
            "require_approval_for_high_impact": True,
            "auto_escalation_threshold": 1000000,  # $1M impact
            "emergency_automation_enabled": True,
            "business_hours_only": False,
            "auto_rollback_on_failure": True,
            "max_retry_attempts": 3,
        }

    async def process_anomaly(
        self,
        anomaly_id: str,
        anomaly_data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> DecisionRecommendation:
        """Process anomaly and generate real-time decision recommendation."""
        context = context or {}

        # Calculate business impact
        business_impact = await self.business_impact_analyzer.analyze_anomaly_impact(
            anomaly_id, anomaly_data, context
        )

        # Create decision context
        decision_context = DecisionContext(
            anomaly_id=anomaly_id,
            timestamp=datetime.now(),
            anomaly_data=anomaly_data,
            business_impact=business_impact,
            system_state=context.get("system_state", {}),
            user_preferences=context.get("user_preferences", {}),
            organizational_policies=context.get("organizational_policies", {}),
            resource_availability=context.get("resource_availability", {}),
        )

        # Generate recommendation
        recommendation = self.decision_engine.generate_recommendation(decision_context)

        if recommendation:
            # Store active decision
            self.active_decisions[anomaly_id] = recommendation

            # Check if auto-execution is appropriate
            if self._should_auto_execute(recommendation):
                await self._queue_for_execution(recommendation, decision_context)

        return recommendation

    def _should_auto_execute(self, recommendation: DecisionRecommendation) -> bool:
        """Determine if recommendation should be auto-executed."""
        # Check automation policies
        if not self.automation_policies.get("emergency_automation_enabled", False):
            if recommendation.time_sensitivity == "immediate":
                return False

        # Check business hours constraint
        if self.automation_policies.get("business_hours_only", False):
            current_hour = datetime.now().hour
            if not (9 <= current_hour <= 17):  # Outside business hours
                return False

        # Check impact threshold
        if recommendation.estimated_impact > self.automation_policies.get(
            "auto_escalation_threshold", 1000000
        ):
            if self.automation_policies.get("require_approval_for_high_impact", True):
                return False

        # Check confidence level
        if recommendation.confidence in [ConfidenceLevel.VERY_LOW, ConfidenceLevel.LOW]:
            return False

        # Check if any actions require approval
        for action in recommendation.recommended_actions:
            if action.requires_approval:
                return False

        return True

    async def _queue_for_execution(
        self, recommendation: DecisionRecommendation, context: DecisionContext
    ) -> None:
        """Queue recommendation for automated execution."""
        execution_item = {
            "recommendation": recommendation,
            "context": context,
            "queued_at": datetime.now(),
            "priority": self._calculate_execution_priority(recommendation),
        }

        self.execution_queue.append(execution_item)

        # Trigger execution if not at capacity
        current_executions = len(
            [item for item in self.execution_queue if item.get("status") == "executing"]
        )

        if current_executions < self.automation_policies.get(
            "max_concurrent_auto_actions", 5
        ):
            await self._execute_next_in_queue()

    def _calculate_execution_priority(
        self, recommendation: DecisionRecommendation
    ) -> int:
        """Calculate execution priority (higher = more urgent)."""
        priority = 0

        # Time sensitivity
        if recommendation.time_sensitivity == "immediate":
            priority += 100
        elif recommendation.time_sensitivity == "urgent":
            priority += 50
        elif recommendation.time_sensitivity == "normal":
            priority += 25

        # Confidence level
        confidence_scores = {
            ConfidenceLevel.VERY_HIGH: 50,
            ConfidenceLevel.HIGH: 40,
            ConfidenceLevel.MEDIUM: 30,
            ConfidenceLevel.LOW: 20,
            ConfidenceLevel.VERY_LOW: 10,
        }
        priority += confidence_scores.get(recommendation.confidence, 0)

        # Decision type urgency
        decision_urgency = {
            DecisionType.QUARANTINE: 90,
            DecisionType.ESCALATE: 70,
            DecisionType.MITIGATE: 60,
            DecisionType.AUTO_REMEDIATE: 50,
            DecisionType.INVESTIGATE: 30,
            DecisionType.ALERT: 20,
            DecisionType.MANUAL_REVIEW: 15,
            DecisionType.IGNORE: 5,
        }
        priority += decision_urgency.get(recommendation.decision, 0)

        return priority

    async def _execute_next_in_queue(self) -> ExecutionResult | None:
        """Execute the next highest priority item in the queue."""
        if not self.execution_queue:
            return None

        # Sort by priority (highest first)
        self.execution_queue = deque(
            sorted(self.execution_queue, key=lambda x: x["priority"], reverse=True)
        )

        execution_item = self.execution_queue.popleft()
        execution_item["status"] = "executing"

        try:
            result = await self._execute_recommendation(
                execution_item["recommendation"], execution_item["context"]
            )
            return result
        except Exception as e:
            logger.error(f"Failed to execute recommendation: {e}")
            return ExecutionResult(
                execution_id=f"exec_{int(datetime.now().timestamp())}",
                status=ResponseStatus.FAILED,
                start_time=datetime.now(),
                end_time=datetime.now(),
                success=False,
                error_message=str(e),
            )

    async def _execute_recommendation(
        self, recommendation: DecisionRecommendation, context: DecisionContext
    ) -> ExecutionResult:
        """Execute a decision recommendation."""
        execution_id = (
            f"exec_{recommendation.recommendation_id}_{int(datetime.now().timestamp())}"
        )
        start_time = datetime.now()

        logger.info(
            f"Executing recommendation {recommendation.recommendation_id}: {recommendation.decision}"
        )

        execution_results = []
        overall_success = True

        # Execute each recommended action
        for action in recommendation.recommended_actions:
            try:
                action_result = await self._execute_action(action, context)
                execution_results.append(action_result)
                if not action_result.success:
                    overall_success = False
            except Exception as e:
                logger.error(f"Failed to execute action {action.action_id}: {e}")
                overall_success = False
                execution_results.append(
                    {"action_id": action.action_id, "success": False, "error": str(e)}
                )

        end_time = datetime.now()

        result = ExecutionResult(
            execution_id=execution_id,
            status=ResponseStatus.COMPLETED
            if overall_success
            else ResponseStatus.FAILED,
            start_time=start_time,
            end_time=end_time,
            success=overall_success,
            output_data={
                "recommendation_id": recommendation.recommendation_id,
                "decision": recommendation.decision.value,
                "actions_executed": len(execution_results),
                "action_results": execution_results,
            },
            measurements={
                "execution_time_seconds": (end_time - start_time).total_seconds(),
                "actions_successful": sum(
                    1 for r in execution_results if r.get("success", False)
                ),
                "actions_failed": sum(
                    1 for r in execution_results if not r.get("success", True)
                ),
            },
        )

        # Store execution result
        self.decision_engine.execution_history.append(result)

        # Update rule performance
        rule_id = recommendation.evidence.get("rule_id")
        if rule_id:
            self._update_rule_performance(rule_id, overall_success)

        return result

    async def _execute_action(
        self, action: ResponseAction, context: DecisionContext
    ) -> dict[str, Any]:
        """Execute a single response action."""
        logger.info(f"Executing action: {action.action_id} - {action.description}")

        # Simulate action execution (in real implementation, this would call actual systems)
        await asyncio.sleep(0.1)  # Simulate processing time

        # Mock success based on action type (in real implementation, this would be actual results)
        success_rate = {
            "notification": 0.95,
            "network_security": 0.90,
            "system_security": 0.85,
            "capacity_management": 0.88,
            "data_protection": 0.92,
        }.get(action.action_type, 0.80)

        success = np.random.random() < success_rate

        return {
            "action_id": action.action_id,
            "action_type": action.action_type,
            "success": success,
            "execution_time_ms": np.random.randint(100, 1000),
            "target_systems": action.target_systems,
            "parameters": action.parameters,
        }

    def _update_rule_performance(self, rule_id: str, success: bool) -> None:
        """Update performance measurements for a rule."""
        if rule_id not in self.decision_engine.rule_performance:
            self.decision_engine.rule_performance[rule_id] = {
                "total_executions": 0,
                "successful_executions": 0,
                "success_rate": 0.0,
            }

        perf = self.decision_engine.rule_performance[rule_id]
        perf["total_executions"] += 1
        if success:
            perf["successful_executions"] += 1
        perf["success_rate"] = perf["successful_executions"] / perf["total_executions"]

    async def get_active_decisions(self) -> list[DecisionRecommendation]:
        """Get all active decision recommendations."""
        return list(self.active_decisions.values())

    async def get_execution_status(self, execution_id: str) -> ExecutionResult | None:
        """Get status of a specific execution."""
        for result in self.decision_engine.execution_history:
            if result.execution_id == execution_id:
                return result
        return None

    async def get_system_measurements(self) -> dict[str, Any]:
        """Get system performance measurements."""
        recent_executions = [
            result
            for result in self.decision_engine.execution_history
            if result.start_time > datetime.now() - timedelta(hours=24)
        ]

        if not recent_executions:
            return {"total_executions": 0, "success_rate": 0.0}

        successful = sum(1 for result in recent_executions if result.success)

        return {
            "total_executions_24h": len(recent_executions),
            "successful_executions_24h": successful,
            "success_rate_24h": successful / len(recent_executions),
            "average_execution_time_seconds": np.mean(
                [
                    result.measurements.get("execution_time_seconds", 0)
                    for result in recent_executions
                ]
            ),
            "queue_length": len(self.execution_queue),
            "active_decisions": len(self.active_decisions),
            "rule_performance": dict(self.decision_engine.rule_performance),
        }
