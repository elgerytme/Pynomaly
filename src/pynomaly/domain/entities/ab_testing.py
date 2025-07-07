"""Domain entities for A/B testing framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

import numpy as np


class TestType(Enum):
    """Type of A/B test."""

    MODEL_COMPARISON = "model_comparison"
    ALGORITHM_COMPARISON = "algorithm_comparison"
    HYPERPARAMETER_OPTIMIZATION = "hyperparameter_optimization"
    FEATURE_ENGINEERING = "feature_engineering"
    ENSEMBLE_CONFIGURATION = "ensemble_configuration"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"


class TestPhase(Enum):
    """Phase of A/B test lifecycle."""

    DESIGN = "design"
    PLANNING = "planning"
    VALIDATION = "validation"
    EXECUTION = "execution"
    MONITORING = "monitoring"
    ANALYSIS = "analysis"
    DECISION = "decision"
    DEPLOYMENT = "deployment"
    POST_DEPLOYMENT = "post_deployment"


class SignificanceTestType(Enum):
    """Type of statistical significance test."""

    TWO_SAMPLE_T_TEST = "two_sample_t_test"
    WELCH_T_TEST = "welch_t_test"
    MANN_WHITNEY_U = "mann_whitney_u"
    WILCOXON_SIGNED_RANK = "wilcoxon_signed_rank"
    CHI_SQUARE = "chi_square"
    FISHER_EXACT = "fisher_exact"
    MCNEMAR_TEST = "mcnemar_test"
    BOOTSTRAP_TEST = "bootstrap_test"
    PERMUTATION_TEST = "permutation_test"
    BAYESIAN_TEST = "bayesian_test"


class EffectSizeMeasure(Enum):
    """Type of effect size measurement."""

    COHENS_D = "cohens_d"
    HEDGES_G = "hedges_g"
    GLASS_DELTA = "glass_delta"
    CLIFF_DELTA = "cliff_delta"
    PROBABILITY_SUPERIORITY = "probability_superiority"
    COMMON_LANGUAGE_EFFECT = "common_language_effect"
    RELATIVE_RISK = "relative_risk"
    ODDS_RATIO = "odds_ratio"


@dataclass
class HypothesisTest:
    """Statistical hypothesis test configuration."""

    test_id: UUID = field(default_factory=uuid4)
    test_type: SignificanceTestType = SignificanceTestType.TWO_SAMPLE_T_TEST
    null_hypothesis: str = ""
    alternative_hypothesis: str = ""
    significance_level: float = 0.05
    power: float = 0.8
    effect_size_measure: EffectSizeMeasure = EffectSizeMeasure.COHENS_D
    minimum_detectable_effect: float = 0.1
    two_tailed: bool = True
    assumptions: list[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate hypothesis test configuration."""
        if not (0.0 < self.significance_level < 1.0):
            raise ValueError("Significance level must be between 0.0 and 1.0")
        if not (0.0 < self.power < 1.0):
            raise ValueError("Power must be between 0.0 and 1.0")
        if self.minimum_detectable_effect <= 0.0:
            raise ValueError("Minimum detectable effect must be positive")

    def get_critical_value(self) -> float:
        """Get critical value for the test."""
        # This would implement critical value calculation based on test type
        return 1.96  # Placeholder for z-test critical value at α=0.05


@dataclass
class ExperimentalDesign:
    """Experimental design configuration for A/B tests."""

    design_id: UUID = field(default_factory=uuid4)
    design_type: str = "randomized_controlled_trial"
    randomization_strategy: str = "simple_randomization"
    stratification_variables: list[str] = field(default_factory=list)
    blocking_variables: list[str] = field(default_factory=list)
    covariate_adjustment: bool = False
    crossover_design: bool = False
    factorial_design: bool = False
    sample_size_calculation: dict[str, Any] = field(default_factory=dict)
    allocation_ratio: tuple[float, ...] = field(default_factory=lambda: (0.5, 0.5))

    def __post_init__(self):
        """Validate experimental design."""
        if abs(sum(self.allocation_ratio) - 1.0) > 1e-6:
            raise ValueError("Allocation ratios must sum to 1.0")
        if len(self.allocation_ratio) < 2:
            raise ValueError("Must have at least 2 allocation groups")

    def get_allocation_for_variant(self, variant_index: int) -> float:
        """Get allocation ratio for specific variant."""
        if 0 <= variant_index < len(self.allocation_ratio):
            return self.allocation_ratio[variant_index]
        return 0.0

    def is_balanced_design(self) -> bool:
        """Check if design has balanced allocation."""
        return len(set(self.allocation_ratio)) == 1


@dataclass
class PowerAnalysis:
    """Statistical power analysis for sample size determination."""

    analysis_id: UUID = field(default_factory=uuid4)
    test_type: SignificanceTestType = SignificanceTestType.TWO_SAMPLE_T_TEST
    effect_size: float = 0.0
    alpha: float = 0.05
    power: float = 0.8
    sample_size_per_group: int = 0
    total_sample_size: int = 0
    allocation_ratio: float = 1.0
    variance_estimate: float | None = None
    baseline_rate: float | None = None

    def __post_init__(self):
        """Validate and compute power analysis."""
        if self.effect_size <= 0.0:
            raise ValueError("Effect size must be positive")
        if not (0.0 < self.alpha < 1.0):
            raise ValueError("Alpha must be between 0.0 and 1.0")
        if not (0.0 < self.power < 1.0):
            raise ValueError("Power must be between 0.0 and 1.0")

        # Calculate sample size if not provided
        if self.sample_size_per_group == 0:
            self.sample_size_per_group = self._calculate_sample_size()
            self.total_sample_size = int(
                self.sample_size_per_group * (1 + self.allocation_ratio)
            )

    def _calculate_sample_size(self) -> int:
        """Calculate required sample size per group."""
        # Simplified sample size calculation for two-sample t-test
        # In practice, would use proper statistical formulas
        from scipy import stats

        z_alpha = stats.norm.ppf(1 - self.alpha / 2)
        z_beta = stats.norm.ppf(self.power)

        # Cohen's formula
        n = 2 * ((z_alpha + z_beta) / self.effect_size) ** 2

        return max(int(np.ceil(n)), 30)  # Minimum of 30 per group

    def get_achieved_power(self, actual_sample_size: int) -> float:
        """Calculate achieved power given actual sample size."""
        if actual_sample_size <= 0:
            return 0.0

        # Simplified power calculation
        from scipy import stats

        z_alpha = stats.norm.ppf(1 - self.alpha / 2)
        ncp = self.effect_size * np.sqrt(actual_sample_size / 2)

        power = 1 - stats.norm.cdf(z_alpha - ncp) + stats.norm.cdf(-z_alpha - ncp)
        return float(np.clip(power, 0.0, 1.0))

    def update_with_observed_variance(self, observed_variance: float) -> None:
        """Update analysis with observed variance."""
        self.variance_estimate = observed_variance
        # Recalculate sample size with observed variance
        if observed_variance > 0:
            adjustment_factor = np.sqrt(observed_variance)
            self.sample_size_per_group = int(
                self.sample_size_per_group * adjustment_factor
            )
            self.total_sample_size = int(
                self.sample_size_per_group * (1 + self.allocation_ratio)
            )


@dataclass
class MetricDefinition:
    """Definition of a metric for A/B testing."""

    metric_id: UUID = field(default_factory=uuid4)
    metric_name: str = ""
    metric_type: str = "continuous"  # continuous, binary, count, time_to_event
    description: str = ""
    calculation_method: str = ""
    aggregation_method: str = "mean"  # mean, median, sum, rate, proportion
    is_primary: bool = False
    is_guardrail: bool = False
    direction_of_improvement: str = "higher"  # higher, lower, no_change
    minimum_change_threshold: float = 0.0
    practical_significance_threshold: float = 0.0
    units: str = ""
    transformation: str | None = None  # log, sqrt, inverse, etc.

    def __post_init__(self):
        """Validate metric definition."""
        if not self.metric_name:
            raise ValueError("Metric name cannot be empty")
        if self.metric_type not in ["continuous", "binary", "count", "time_to_event"]:
            raise ValueError("Invalid metric type")
        if self.aggregation_method not in [
            "mean",
            "median",
            "sum",
            "rate",
            "proportion",
        ]:
            raise ValueError("Invalid aggregation method")
        if self.direction_of_improvement not in ["higher", "lower", "no_change"]:
            raise ValueError("Invalid direction of improvement")

    def is_improvement(self, baseline_value: float, test_value: float) -> bool:
        """Check if test value represents an improvement over baseline."""
        if self.direction_of_improvement == "higher":
            return test_value > baseline_value + self.minimum_change_threshold
        elif self.direction_of_improvement == "lower":
            return test_value < baseline_value - self.minimum_change_threshold
        else:  # no_change
            return abs(test_value - baseline_value) <= self.minimum_change_threshold

    def calculate_improvement_magnitude(
        self, baseline_value: float, test_value: float
    ) -> float:
        """Calculate magnitude of improvement."""
        if baseline_value == 0:
            return 0.0

        if self.direction_of_improvement == "higher":
            return (test_value - baseline_value) / baseline_value
        elif self.direction_of_improvement == "lower":
            return (baseline_value - test_value) / baseline_value
        else:  # no_change
            return 0.0


@dataclass
class TestVariant:
    """Test variant in A/B test."""

    variant_id: UUID = field(default_factory=uuid4)
    variant_name: str = ""
    variant_type: str = "model"  # model, algorithm, configuration, feature_set
    description: str = ""
    configuration: dict[str, Any] = field(default_factory=dict)
    is_control: bool = False
    is_treatment: bool = False
    allocation_percentage: float = 0.0
    minimum_sample_size: int = 100
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate test variant."""
        if not self.variant_name:
            self.variant_name = f"variant_{str(self.variant_id)[:8]}"
        if not (0.0 <= self.allocation_percentage <= 100.0):
            raise ValueError("Allocation percentage must be between 0.0 and 100.0")
        if self.minimum_sample_size < 0:
            raise ValueError("Minimum sample size must be non-negative")

    def get_allocation_fraction(self) -> float:
        """Get allocation as fraction (0.0 to 1.0)."""
        return self.allocation_percentage / 100.0

    def set_as_control(self) -> None:
        """Set this variant as the control group."""
        self.is_control = True
        self.is_treatment = False

    def set_as_treatment(self) -> None:
        """Set this variant as a treatment group."""
        self.is_control = False
        self.is_treatment = True


@dataclass
class ObservationUnit:
    """Unit of observation in A/B test."""

    unit_id: UUID = field(default_factory=uuid4)
    unit_type: str = "request"  # request, user, session, transaction
    variant_id: UUID = field(default_factory=uuid4)
    assignment_timestamp: datetime = field(default_factory=datetime.utcnow)
    exposure_timestamp: datetime | None = None
    outcome_timestamp: datetime | None = None
    stratification_variables: dict[str, Any] = field(default_factory=dict)
    covariates: dict[str, Any] = field(default_factory=dict)
    outcomes: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate observation unit."""
        if self.assignment_timestamp and self.exposure_timestamp:
            if self.exposure_timestamp < self.assignment_timestamp:
                raise ValueError(
                    "Exposure timestamp cannot be before assignment timestamp"
                )
        if self.exposure_timestamp and self.outcome_timestamp:
            if self.outcome_timestamp < self.exposure_timestamp:
                raise ValueError(
                    "Outcome timestamp cannot be before exposure timestamp"
                )

    def get_exposure_delay(self) -> timedelta | None:
        """Get delay between assignment and exposure."""
        if self.assignment_timestamp and self.exposure_timestamp:
            return self.exposure_timestamp - self.assignment_timestamp
        return None

    def get_outcome_delay(self) -> timedelta | None:
        """Get delay between exposure and outcome."""
        if self.exposure_timestamp and self.outcome_timestamp:
            return self.outcome_timestamp - self.exposure_timestamp
        return None

    def has_outcome(self, metric_name: str) -> bool:
        """Check if unit has outcome for specific metric."""
        return metric_name in self.outcomes

    def get_outcome_value(self, metric_name: str) -> Any:
        """Get outcome value for specific metric."""
        return self.outcomes.get(metric_name)


@dataclass
class VariantPerformance:
    """Performance metrics for a test variant."""

    variant_id: UUID
    variant_name: str
    sample_size: int = 0
    metric_values: dict[str, list[float]] = field(default_factory=dict)
    aggregated_metrics: dict[str, float] = field(default_factory=dict)
    confidence_intervals: dict[str, tuple[float, float]] = field(default_factory=dict)
    standard_errors: dict[str, float] = field(default_factory=dict)
    percentiles: dict[str, dict[str, float]] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.utcnow)

    def add_observation(self, metric_name: str, value: float) -> None:
        """Add metric observation."""
        if metric_name not in self.metric_values:
            self.metric_values[metric_name] = []

        self.metric_values[metric_name].append(value)
        self.sample_size = max(self.sample_size, len(self.metric_values[metric_name]))
        self.last_updated = datetime.utcnow()

    def calculate_aggregated_metrics(self, metrics: list[MetricDefinition]) -> None:
        """Calculate aggregated metrics."""
        for metric in metrics:
            if metric.metric_name in self.metric_values:
                values = np.array(self.metric_values[metric.metric_name])

                if metric.aggregation_method == "mean":
                    self.aggregated_metrics[metric.metric_name] = float(np.mean(values))
                elif metric.aggregation_method == "median":
                    self.aggregated_metrics[metric.metric_name] = float(
                        np.median(values)
                    )
                elif metric.aggregation_method == "sum":
                    self.aggregated_metrics[metric.metric_name] = float(np.sum(values))
                elif metric.aggregation_method == "rate":
                    self.aggregated_metrics[metric.metric_name] = float(np.mean(values))
                elif metric.aggregation_method == "proportion":
                    self.aggregated_metrics[metric.metric_name] = float(np.mean(values))

                # Calculate confidence interval
                if len(values) > 1:
                    mean_val = np.mean(values)
                    std_err = np.std(values, ddof=1) / np.sqrt(len(values))
                    ci_margin = 1.96 * std_err  # 95% CI

                    self.confidence_intervals[metric.metric_name] = (
                        float(mean_val - ci_margin),
                        float(mean_val + ci_margin),
                    )
                    self.standard_errors[metric.metric_name] = float(std_err)

                    # Calculate percentiles
                    self.percentiles[metric.metric_name] = {
                        "p5": float(np.percentile(values, 5)),
                        "p25": float(np.percentile(values, 25)),
                        "p50": float(np.percentile(values, 50)),
                        "p75": float(np.percentile(values, 75)),
                        "p95": float(np.percentile(values, 95)),
                    }

    def get_metric_summary(self, metric_name: str) -> dict[str, Any]:
        """Get comprehensive summary for a metric."""
        if metric_name not in self.metric_values:
            return {}

        return {
            "sample_size": len(self.metric_values[metric_name]),
            "aggregated_value": self.aggregated_metrics.get(metric_name),
            "confidence_interval": self.confidence_intervals.get(metric_name),
            "standard_error": self.standard_errors.get(metric_name),
            "percentiles": self.percentiles.get(metric_name, {}),
            "raw_values": self.metric_values[metric_name][:100],  # Limit for display
        }


@dataclass
class ComparisonResult:
    """Result of comparing two variants."""

    comparison_id: UUID = field(default_factory=uuid4)
    control_variant_id: UUID = field(default_factory=uuid4)
    treatment_variant_id: UUID = field(default_factory=uuid4)
    metric_name: str = ""
    test_type: SignificanceTestType = SignificanceTestType.TWO_SAMPLE_T_TEST
    control_value: float = 0.0
    treatment_value: float = 0.0
    difference: float = 0.0
    relative_difference: float = 0.0
    effect_size: float = 0.0
    effect_size_measure: EffectSizeMeasure = EffectSizeMeasure.COHENS_D
    test_statistic: float = 0.0
    p_value: float = 1.0
    confidence_interval: tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))
    is_significant: bool = False
    power: float = 0.0
    interpretation: str = ""
    practical_significance: bool = False

    def __post_init__(self):
        """Validate comparison result."""
        if not (0.0 <= self.p_value <= 1.0):
            raise ValueError("P-value must be between 0.0 and 1.0")
        if not (0.0 <= self.power <= 1.0):
            raise ValueError("Power must be between 0.0 and 1.0")

        # Calculate relative difference
        if self.control_value != 0:
            self.relative_difference = (
                self.treatment_value - self.control_value
            ) / self.control_value

        # Calculate absolute difference
        self.difference = self.treatment_value - self.control_value

    def get_effect_interpretation(self) -> str:
        """Get interpretation of effect size."""
        if self.effect_size_measure == EffectSizeMeasure.COHENS_D:
            if abs(self.effect_size) < 0.2:
                return "negligible"
            elif abs(self.effect_size) < 0.5:
                return "small"
            elif abs(self.effect_size) < 0.8:
                return "medium"
            else:
                return "large"

        return "unknown"

    def is_practically_significant(self, threshold: float = 0.05) -> bool:
        """Check if result is practically significant."""
        return abs(self.relative_difference) >= threshold

    def get_confidence_level(self) -> float:
        """Get confidence level from p-value."""
        return 1.0 - self.p_value

    def get_result_summary(self) -> dict[str, Any]:
        """Get comprehensive summary of comparison result."""
        return {
            "metric": self.metric_name,
            "control_value": self.control_value,
            "treatment_value": self.treatment_value,
            "absolute_difference": self.difference,
            "relative_difference": self.relative_difference,
            "effect_size": self.effect_size,
            "effect_interpretation": self.get_effect_interpretation(),
            "p_value": self.p_value,
            "is_significant": self.is_significant,
            "practical_significance": self.practical_significance,
            "confidence_interval": self.confidence_interval,
            "power": self.power,
            "interpretation": self.interpretation,
        }


@dataclass
class TestEvidence:
    """Evidence collected during A/B test."""

    evidence_id: UUID = field(default_factory=uuid4)
    test_id: UUID = field(default_factory=uuid4)
    evidence_type: str = (
        "performance_data"  # performance_data, user_feedback, system_logs
    )
    collection_timestamp: datetime = field(default_factory=datetime.utcnow)
    variant_id: UUID = field(default_factory=uuid4)
    evidence_data: dict[str, Any] = field(default_factory=dict)
    quality_score: float = 1.0
    reliability_score: float = 1.0
    bias_indicators: list[str] = field(default_factory=list)
    validation_status: str = "pending"  # pending, validated, rejected

    def __post_init__(self):
        """Validate test evidence."""
        if not (0.0 <= self.quality_score <= 1.0):
            raise ValueError("Quality score must be between 0.0 and 1.0")
        if not (0.0 <= self.reliability_score <= 1.0):
            raise ValueError("Reliability score must be between 0.0 and 1.0")
        if self.validation_status not in ["pending", "validated", "rejected"]:
            raise ValueError("Invalid validation status")

    def is_high_quality(self, threshold: float = 0.8) -> bool:
        """Check if evidence meets high quality threshold."""
        return self.quality_score >= threshold and self.reliability_score >= threshold

    def has_bias_indicators(self) -> bool:
        """Check if evidence has bias indicators."""
        return len(self.bias_indicators) > 0

    def get_credibility_score(self) -> float:
        """Calculate overall credibility score."""
        bias_penalty = len(self.bias_indicators) * 0.1
        credibility = (self.quality_score + self.reliability_score) / 2 - bias_penalty
        return max(0.0, min(1.0, credibility))


@dataclass
class DecisionFramework:
    """Framework for making decisions based on A/B test results."""

    framework_id: UUID = field(default_factory=uuid4)
    decision_criteria: dict[str, Any] = field(default_factory=dict)
    risk_tolerance: float = 0.05
    minimum_practical_effect: float = 0.02
    required_confidence: float = 0.95
    business_constraints: list[str] = field(default_factory=list)
    cost_benefit_analysis: dict[str, float] = field(default_factory=dict)
    stakeholder_weights: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        """Validate decision framework."""
        if not (0.0 < self.risk_tolerance < 1.0):
            raise ValueError("Risk tolerance must be between 0.0 and 1.0")
        if not (0.0 < self.required_confidence < 1.0):
            raise ValueError("Required confidence must be between 0.0 and 1.0")
        if self.minimum_practical_effect < 0.0:
            raise ValueError("Minimum practical effect must be non-negative")

    def evaluate_decision(
        self, comparison_results: list[ComparisonResult]
    ) -> dict[str, Any]:
        """Evaluate decision based on test results."""
        decision = {
            "recommendation": "no_change",
            "confidence": 0.0,
            "rationale": [],
            "risks": [],
            "benefits": [],
        }

        significant_results = [r for r in comparison_results if r.is_significant]
        practically_significant = [
            r
            for r in comparison_results
            if r.is_practically_significant(self.minimum_practical_effect)
        ]

        if significant_results and practically_significant:
            positive_effects = [r for r in practically_significant if r.difference > 0]
            if positive_effects:
                decision["recommendation"] = "deploy_treatment"
                decision["confidence"] = min(
                    [r.get_confidence_level() for r in positive_effects]
                )
                decision["rationale"].append(
                    "Statistically and practically significant improvement"
                )

        return decision

    def assess_deployment_risk(
        self, comparison_results: list[ComparisonResult]
    ) -> float:
        """Assess risk of deploying treatment variant."""
        risk_factors = []

        for result in comparison_results:
            if result.is_significant and result.difference < 0:
                risk_factors.append(abs(result.relative_difference))

        if not risk_factors:
            return 0.0

        return min(1.0, np.mean(risk_factors))


@dataclass
class ABTesting:
    """Complete A/B testing framework for anomaly detection models."""

    test_id: UUID = field(default_factory=uuid4)
    test_name: str = ""
    test_description: str = ""
    test_type: TestType = TestType.MODEL_COMPARISON
    current_phase: TestPhase = TestPhase.DESIGN
    hypothesis_test: HypothesisTest = field(default_factory=HypothesisTest)
    experimental_design: ExperimentalDesign = field(default_factory=ExperimentalDesign)
    power_analysis: PowerAnalysis = field(default_factory=PowerAnalysis)
    metrics: list[MetricDefinition] = field(default_factory=list)
    variants: list[TestVariant] = field(default_factory=list)
    observations: list[ObservationUnit] = field(default_factory=list)
    variant_performances: dict[UUID, VariantPerformance] = field(default_factory=dict)
    comparison_results: list[ComparisonResult] = field(default_factory=list)
    evidence_collection: list[TestEvidence] = field(default_factory=list)
    decision_framework: DecisionFramework = field(default_factory=DecisionFramework)
    start_time: datetime | None = None
    end_time: datetime | None = None
    status: str = "draft"  # draft, active, paused, completed, terminated
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate A/B testing configuration."""
        if not self.test_name:
            self.test_name = f"ab_test_{str(self.test_id)[:8]}"
        
        if self.status not in ["draft", "active", "paused", "completed", "terminated"]:
            raise ValueError("Invalid test status")
        
        # Validate that we have at least one control and one treatment
        if len(self.variants) >= 2:
            control_count = sum(1 for v in self.variants if v.is_control)
            treatment_count = sum(1 for v in self.variants if v.is_treatment)
            
            if control_count == 0:
                raise ValueError("Must have at least one control variant")
            if treatment_count == 0:
                raise ValueError("Must have at least one treatment variant")

    def add_variant(self, variant: TestVariant) -> None:
        """Add a variant to the test."""
        # Check allocation doesn't exceed 100%
        total_allocation = sum(v.allocation_percentage for v in self.variants)
        if total_allocation + variant.allocation_percentage > 100.0:
            raise ValueError("Total allocation would exceed 100%")
        
        self.variants.append(variant)
        
        # Initialize performance tracking
        self.variant_performances[variant.variant_id] = VariantPerformance(
            variant_id=variant.variant_id,
            variant_name=variant.variant_name
        )

    def add_metric(self, metric: MetricDefinition) -> None:
        """Add a metric to track."""
        # Ensure only one primary metric
        if metric.is_primary:
            for existing_metric in self.metrics:
                if existing_metric.is_primary:
                    existing_metric.is_primary = False
        
        self.metrics.append(metric)

    def start_test(self) -> None:
        """Start the A/B test."""
        if self.status != "draft":
            raise ValueError("Can only start test from draft status")
        
        # Validate test is ready
        if len(self.variants) < 2:
            raise ValueError("Need at least 2 variants to start test")
        if len(self.metrics) == 0:
            raise ValueError("Need at least 1 metric to track")
        
        # Check total allocation is 100%
        total_allocation = sum(v.allocation_percentage for v in self.variants)
        if abs(total_allocation - 100.0) > 0.01:
            raise ValueError("Total variant allocation must equal 100%")
        
        self.status = "active"
        self.current_phase = TestPhase.EXECUTION
        self.start_time = datetime.utcnow()

    def pause_test(self) -> None:
        """Pause the A/B test."""
        if self.status != "active":
            raise ValueError("Can only pause active test")
        
        self.status = "paused"

    def resume_test(self) -> None:
        """Resume a paused A/B test."""
        if self.status != "paused":
            raise ValueError("Can only resume paused test")
        
        self.status = "active"

    def complete_test(self) -> None:
        """Complete the A/B test."""
        if self.status not in ["active", "paused"]:
            raise ValueError("Can only complete active or paused test")
        
        self.status = "completed"
        self.current_phase = TestPhase.ANALYSIS
        self.end_time = datetime.utcnow()

    def add_observation(self, observation: ObservationUnit) -> None:
        """Add an observation to the test."""
        if self.status != "active":
            raise ValueError("Can only add observations to active test")
        
        # Validate variant exists
        variant_ids = {v.variant_id for v in self.variants}
        if observation.variant_id not in variant_ids:
            raise ValueError("Observation variant_id not found in test variants")
        
        self.observations.append(observation)
        
        # Update variant performance
        if observation.variant_id in self.variant_performances:
            for metric in self.metrics:
                if observation.has_outcome(metric.metric_name):
                    value = observation.get_outcome_value(metric.metric_name)
                    if isinstance(value, (int, float)):
                        self.variant_performances[observation.variant_id].add_observation(
                            metric.metric_name, float(value)
                        )

    def calculate_results(self) -> None:
        """Calculate test results and comparisons."""
        if len(self.observations) == 0:
            return
        
        # Update variant performance metrics
        for variant_id, performance in self.variant_performances.items():
            performance.calculate_aggregated_metrics(self.metrics)
        
        # Perform statistical comparisons
        control_variants = [v for v in self.variants if v.is_control]
        treatment_variants = [v for v in self.variants if v.is_treatment]
        
        self.comparison_results.clear()
        
        for control in control_variants:
            for treatment in treatment_variants:
                for metric in self.metrics:
                    if (control.variant_id in self.variant_performances and 
                        treatment.variant_id in self.variant_performances):
                        
                        control_perf = self.variant_performances[control.variant_id]
                        treatment_perf = self.variant_performances[treatment.variant_id]
                        
                        if (metric.metric_name in control_perf.aggregated_metrics and
                            metric.metric_name in treatment_perf.aggregated_metrics):
                            
                            result = self._perform_statistical_test(
                                control, treatment, metric, control_perf, treatment_perf
                            )
                            self.comparison_results.append(result)

    def _perform_statistical_test(
        self,
        control_variant: TestVariant,
        treatment_variant: TestVariant,
        metric: MetricDefinition,
        control_performance: VariantPerformance,
        treatment_performance: VariantPerformance
    ) -> ComparisonResult:
        """Perform statistical test between variants."""
        control_value = control_performance.aggregated_metrics[metric.metric_name]
        treatment_value = treatment_performance.aggregated_metrics[metric.metric_name]
        
        # Simple t-test implementation (placeholder)
        # In practice, would use proper statistical libraries
        from scipy import stats
        
        control_values = np.array(control_performance.metric_values[metric.metric_name])
        treatment_values = np.array(treatment_performance.metric_values[metric.metric_name])
        
        if len(control_values) > 1 and len(treatment_values) > 1:
            t_stat, p_value = stats.ttest_ind(treatment_values, control_values)
            
            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt(((len(control_values) - 1) * np.var(control_values, ddof=1) +
                                  (len(treatment_values) - 1) * np.var(treatment_values, ddof=1)) /
                                 (len(control_values) + len(treatment_values) - 2))
            
            effect_size = (treatment_value - control_value) / pooled_std if pooled_std > 0 else 0.0
            
            # Confidence interval for difference
            se_diff = pooled_std * np.sqrt(1/len(control_values) + 1/len(treatment_values))
            ci_margin = 1.96 * se_diff
            ci_lower = (treatment_value - control_value) - ci_margin
            ci_upper = (treatment_value - control_value) + ci_margin
            
        else:
            t_stat, p_value = 0.0, 1.0
            effect_size = 0.0
            ci_lower, ci_upper = 0.0, 0.0
        
        return ComparisonResult(
            control_variant_id=control_variant.variant_id,
            treatment_variant_id=treatment_variant.variant_id,
            metric_name=metric.metric_name,
            test_type=self.hypothesis_test.test_type,
            control_value=control_value,
            treatment_value=treatment_value,
            effect_size=effect_size,
            test_statistic=float(t_stat),
            p_value=float(p_value),
            confidence_interval=(ci_lower, ci_upper),
            is_significant=p_value < self.hypothesis_test.significance_level,
            practical_significance=metric.is_improvement(control_value, treatment_value)
        )

    def get_test_summary(self) -> dict[str, Any]:
        """Get comprehensive test summary."""
        return {
            "test_id": str(self.test_id),
            "test_name": self.test_name,
            "test_type": self.test_type.value,
            "status": self.status,
            "phase": self.current_phase.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": str(self.end_time - self.start_time) if self.start_time and self.end_time else None,
            "variants": len(self.variants),
            "metrics": len(self.metrics),
            "observations": len(self.observations),
            "comparison_results": len(self.comparison_results),
            "significant_results": len([r for r in self.comparison_results if r.is_significant]),
        }

    def get_decision_recommendation(self) -> dict[str, Any]:
        """Get decision recommendation based on test results."""
        if not self.comparison_results:
            return {
                "recommendation": "insufficient_data",
                "rationale": "No comparison results available"
            }
        
        return self.decision_framework.evaluate_decision(self.comparison_results)
