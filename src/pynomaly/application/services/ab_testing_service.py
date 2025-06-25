"""A/B testing service for model performance comparison and validation."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from pynomaly.application.services.auto_retraining_service import AutoRetrainingService
from pynomaly.domain.entities.ab_test import ABTest

logger = logging.getLogger(__name__)


class ABTestStatus(Enum):
    """Status of A/B test."""

    PLANNING = "planning"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    TERMINATED = "terminated"
    FAILED = "failed"


class TestConclusion(Enum):
    """Conclusion of A/B test."""

    CHAMPION_WINS = "champion_wins"
    CHALLENGER_WINS = "challenger_wins"
    NO_SIGNIFICANT_DIFFERENCE = "no_significant_difference"
    INCONCLUSIVE = "inconclusive"
    TEST_INVALID = "test_invalid"


class TrafficSplitStrategy(Enum):
    """Traffic splitting strategy for A/B testing."""

    RANDOM = "random"
    ROUND_ROBIN = "round_robin"
    GEOGRAPHIC = "geographic"
    TIME_BASED = "time_based"
    HASH_BASED = "hash_based"
    GRADUAL_ROLLOUT = "gradual_rollout"


@dataclass
class TestConfiguration:
    """Configuration for A/B test."""

    test_name: str
    description: str
    traffic_split_strategy: TrafficSplitStrategy = TrafficSplitStrategy.RANDOM
    champion_traffic_percentage: float = 0.5
    challenger_traffic_percentage: float = 0.5
    minimum_sample_size: int = 1000
    maximum_duration_days: int = 30
    significance_level: float = 0.05
    minimum_effect_size: float = 0.02
    power_threshold: float = 0.8
    early_stopping_enabled: bool = True
    early_stopping_check_interval_hours: int = 6
    performance_metrics: list[str] = field(
        default_factory=lambda: [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
        ]
    )
    business_metrics: list[str] = field(default_factory=list)
    guardrail_metrics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        """Validate test configuration."""
        if not (0.0 < self.champion_traffic_percentage < 1.0):
            raise ValueError("Champion traffic percentage must be between 0.0 and 1.0")
        if not (0.0 < self.challenger_traffic_percentage < 1.0):
            raise ValueError(
                "Challenger traffic percentage must be between 0.0 and 1.0"
            )
        if (
            abs(
                self.champion_traffic_percentage
                + self.challenger_traffic_percentage
                - 1.0
            )
            > 1e-6
        ):
            raise ValueError("Traffic percentages must sum to 1.0")
        if not (0.0 < self.significance_level < 1.0):
            raise ValueError("Significance level must be between 0.0 and 1.0")
        if not (0.0 < self.power_threshold < 1.0):
            raise ValueError("Power threshold must be between 0.0 and 1.0")


@dataclass
class ModelVariant:
    """Model variant in A/B test."""

    variant_id: UUID = field(default_factory=uuid4)
    variant_name: str = ""
    model: BaseEstimator | None = None
    model_version: str = "1.0.0"
    deployment_timestamp: datetime = field(default_factory=datetime.utcnow)
    traffic_allocation: float = 0.5
    is_champion: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate model variant."""
        if not (0.0 <= self.traffic_allocation <= 1.0):
            raise ValueError("Traffic allocation must be between 0.0 and 1.0")
        if not self.variant_name:
            self.variant_name = f"variant_{str(self.variant_id)[:8]}"


@dataclass
class PredictionRecord:
    """Record of a single prediction during A/B test."""

    record_id: UUID = field(default_factory=uuid4)
    variant_id: UUID = field(default_factory=uuid4)
    prediction_timestamp: datetime = field(default_factory=datetime.utcnow)
    input_features: np.ndarray | None = None
    prediction: Any = None
    prediction_confidence: float = 0.0
    actual_label: Any | None = None
    feedback_received: bool = False
    processing_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ABTestMetrics:
    """Metrics collected during A/B test."""

    variant_id: UUID
    variant_name: str
    total_predictions: int = 0
    correct_predictions: int = 0
    performance_metrics: dict[str, float] = field(default_factory=dict)
    business_metrics: dict[str, float] = field(default_factory=dict)
    confidence_intervals: dict[str, tuple[float, float]] = field(default_factory=dict)
    statistical_tests: dict[str, dict[str, float]] = field(default_factory=dict)
    processing_times: list[float] = field(default_factory=list)
    error_rate: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)

    def get_accuracy(self) -> float:
        """Get accuracy metric."""
        if self.total_predictions == 0:
            return 0.0
        return self.correct_predictions / self.total_predictions

    def get_average_processing_time(self) -> float:
        """Get average processing time."""
        if not self.processing_times:
            return 0.0
        return float(np.mean(self.processing_times))

    def get_p99_processing_time(self) -> float:
        """Get 99th percentile processing time."""
        if not self.processing_times:
            return 0.0
        return float(np.percentile(self.processing_times, 99))


@dataclass
class StatisticalTestResult:
    """Result of statistical test comparing variants."""

    test_name: str
    metric_name: str
    champion_value: float
    challenger_value: float
    test_statistic: float
    p_value: float
    effect_size: float
    confidence_interval: tuple[float, float]
    is_significant: bool
    power: float
    interpretation: str

    def get_result_summary(self) -> dict[str, Any]:
        """Get summary of test result."""
        return {
            "test": self.test_name,
            "metric": self.metric_name,
            "champion": self.champion_value,
            "challenger": self.challenger_value,
            "p_value": self.p_value,
            "effect_size": self.effect_size,
            "significant": self.is_significant,
            "power": self.power,
            "interpretation": self.interpretation,
        }


@dataclass
class ABTestResult:
    """Final result of A/B test."""

    test_id: UUID
    test_name: str
    conclusion: TestConclusion
    confidence: float
    champion_metrics: ABTestMetrics
    challenger_metrics: ABTestMetrics
    statistical_tests: list[StatisticalTestResult]
    duration: timedelta
    total_samples: int
    recommendation: str
    risk_assessment: dict[str, float]
    business_impact_estimate: dict[str, float] = field(default_factory=dict)

    def get_winning_variant(self) -> UUID | None:
        """Get ID of winning variant."""
        if self.conclusion == TestConclusion.CHAMPION_WINS:
            return self.champion_metrics.variant_id
        elif self.conclusion == TestConclusion.CHALLENGER_WINS:
            return self.challenger_metrics.variant_id
        return None

    def is_actionable(self) -> bool:
        """Check if test result is actionable."""
        return (
            self.conclusion
            in [TestConclusion.CHAMPION_WINS, TestConclusion.CHALLENGER_WINS]
            and self.confidence > 0.8
        )


class ABTestingService:
    """Service for conducting A/B tests on model variants.

    This service provides comprehensive A/B testing capabilities including:
    - Statistical test design and sample size calculation
    - Traffic splitting and variant allocation
    - Real-time performance monitoring
    - Statistical significance testing
    - Early stopping and guardrail monitoring
    - Comprehensive reporting and recommendations
    """

    def __init__(
        self,
        storage_path: Path,
        auto_retraining_service: AutoRetrainingService | None = None,
    ):
        """Initialize A/B testing service.

        Args:
            storage_path: Path for storing test data and results
            auto_retraining_service: Auto-retraining service for integration
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.auto_retraining_service = auto_retraining_service

        # Active tests
        self.active_tests: dict[UUID, ABTest] = {}

        # Prediction records
        self.prediction_records: dict[UUID, list[PredictionRecord]] = {}

        # Traffic router
        self.traffic_router: TrafficRouter = None  # Will be initialized later

        # Statistical analyzer
        self.statistical_analyzer: StatisticalAnalyzer = None  # Will be initialized later

        # Guardrail monitor
        self.guardrail_monitor: GuardrailMonitor = None  # Will be initialized later

        # Background tasks
        self._background_tasks: list[asyncio.Task] = []

        # Initialize components after class definitions are available
        self._initialize_components()

        # Load existing tests
        asyncio.create_task(self._load_active_tests())

    def _initialize_components(self) -> None:
        """Initialize service components."""
        self.traffic_router = TrafficRouter()
        self.statistical_analyzer = StatisticalAnalyzer()
        self.guardrail_monitor = GuardrailMonitor()

    async def _load_active_tests(self) -> None:
        """Load active tests from storage."""
        test_files = self.storage_path.glob("test_*.json")

        for test_file in test_files:
            try:
                test = await self._load_test_from_file(test_file)
                if test.status in [ABTestStatus.RUNNING, ABTestStatus.PAUSED]:
                    self.active_tests[test.test_id] = test
                    logger.info(f"Loaded active A/B test {test.test_id}")
            except Exception as e:
                logger.warning(f"Failed to load test from {test_file}: {e}")

    async def design_ab_test(
        self,
        champion_model: BaseEstimator,
        challenger_model: BaseEstimator,
        test_config: TestConfiguration,
    ) -> UUID:
        """Design and initialize A/B test.

        Args:
            champion_model: Current production model (champion)
            challenger_model: New model to test (challenger)
            test_config: Test configuration

        Returns:
            Test ID
        """
        try:
            # Create model variants
            champion_variant = ModelVariant(
                variant_name="champion",
                model=champion_model,
                traffic_allocation=test_config.champion_traffic_percentage,
                is_champion=True,
                metadata={
                    "role": "champion",
                    "description": "Current production model",
                },
            )

            challenger_variant = ModelVariant(
                variant_name="challenger",
                model=challenger_model,
                traffic_allocation=test_config.challenger_traffic_percentage,
                is_champion=False,
                metadata={"role": "challenger", "description": "New model candidate"},
            )

            # Calculate required sample size
            required_sample_size = self._calculate_sample_size(test_config)

            # Create A/B test
            ab_test = ABTest(
                test_name=test_config.test_name,
                description=test_config.description,
                configuration=test_config,
                champion_variant=champion_variant,
                challenger_variant=challenger_variant,
                required_sample_size=max(
                    required_sample_size, test_config.minimum_sample_size
                ),
            )

            # Initialize metrics
            self.prediction_records[ab_test.test_id] = []

            # Store test
            self.active_tests[ab_test.test_id] = ab_test
            await self._save_test(ab_test)

            logger.info(f"Designed A/B test {ab_test.test_id}: {test_config.test_name}")
            return ab_test.test_id

        except Exception as e:
            logger.error(f"Failed to design A/B test: {e}")
            raise

    async def start_ab_test(self, test_id: UUID) -> bool:
        """Start an A/B test.

        Args:
            test_id: Test ID

        Returns:
            True if successfully started
        """
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")

        test = self.active_tests[test_id]

        if test.status != ABTestStatus.PLANNING:
            raise ValueError(f"Test {test_id} is not in planning state")

        try:
            # Start the test
            test.status = ABTestStatus.RUNNING
            test.started_at = datetime.utcnow()

            # Configure traffic router
            await self.traffic_router.configure_test(test)

            # Start background monitoring
            if test.configuration.early_stopping_enabled:
                task = asyncio.create_task(self._monitor_test(test_id))
                self._background_tasks.append(task)

            # Save updated test
            await self._save_test(test)

            logger.info(f"Started A/B test {test_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to start A/B test {test_id}: {e}")
            test.status = ABTestStatus.FAILED
            test.error_message = str(e)
            await self._save_test(test)
            return False

    async def route_prediction(
        self,
        test_id: UUID,
        input_features: np.ndarray,
        request_metadata: dict[str, Any] | None = None,
    ) -> tuple[Any, UUID]:
        """Route prediction request to appropriate model variant.

        Args:
            test_id: Test ID
            input_features: Input features for prediction
            request_metadata: Optional request metadata

        Returns:
            Tuple of (prediction, variant_id)
        """
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")

        test = self.active_tests[test_id]

        if test.status != ABTestStatus.RUNNING:
            raise ValueError(f"Test {test_id} is not running")

        try:
            start_time = datetime.utcnow()

            # Route to variant
            variant = await self.traffic_router.route_request(test, request_metadata)

            # Make prediction
            prediction = variant.model.predict(input_features.reshape(1, -1))[0]

            # Calculate confidence if available
            confidence = 0.0
            if hasattr(variant.model, "predict_proba"):
                proba = variant.model.predict_proba(input_features.reshape(1, -1))[0]
                confidence = float(np.max(proba))
            elif hasattr(variant.model, "decision_function"):
                decision = variant.model.decision_function(
                    input_features.reshape(1, -1)
                )[0]
                confidence = float(abs(decision))

            # Record prediction
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            record = PredictionRecord(
                variant_id=variant.variant_id,
                input_features=input_features,
                prediction=prediction,
                prediction_confidence=confidence,
                processing_time_ms=processing_time,
                metadata=request_metadata or {},
            )

            self.prediction_records[test_id].append(record)

            return prediction, variant.variant_id

        except Exception as e:
            logger.error(f"Failed to route prediction for test {test_id}: {e}")
            raise

    async def record_feedback(
        self,
        test_id: UUID,
        prediction_record_id: UUID,
        actual_label: Any,
        feedback_metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Record actual label/feedback for a prediction.

        Args:
            test_id: Test ID
            prediction_record_id: Prediction record ID
            actual_label: Actual label/outcome
            feedback_metadata: Optional feedback metadata

        Returns:
            True if successfully recorded
        """
        if test_id not in self.prediction_records:
            return False

        try:
            # Find prediction record
            for record in self.prediction_records[test_id]:
                if record.record_id == prediction_record_id:
                    record.actual_label = actual_label
                    record.feedback_received = True
                    if feedback_metadata:
                        record.metadata.update(feedback_metadata)

                    logger.debug(
                        f"Recorded feedback for prediction {prediction_record_id}"
                    )
                    return True

            return False

        except Exception as e:
            logger.error(f"Failed to record feedback: {e}")
            return False

    async def analyze_test_performance(self, test_id: UUID) -> dict[str, ABTestMetrics]:
        """Analyze current performance of A/B test variants.

        Args:
            test_id: Test ID

        Returns:
            Dictionary of variant metrics
        """
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")

        test = self.active_tests[test_id]
        records = self.prediction_records.get(test_id, [])

        try:
            # Separate records by variant
            champion_records = [
                r for r in records if r.variant_id == test.champion_variant.variant_id
            ]
            challenger_records = [
                r for r in records if r.variant_id == test.challenger_variant.variant_id
            ]

            # Calculate metrics for each variant
            champion_metrics = await self._calculate_variant_metrics(
                test.champion_variant, champion_records, test.configuration
            )
            challenger_metrics = await self._calculate_variant_metrics(
                test.challenger_variant, challenger_records, test.configuration
            )

            return {"champion": champion_metrics, "challenger": challenger_metrics}

        except Exception as e:
            logger.error(f"Failed to analyze test performance for {test_id}: {e}")
            raise

    async def check_statistical_significance(
        self, test_id: UUID
    ) -> list[StatisticalTestResult]:
        """Check statistical significance of A/B test results.

        Args:
            test_id: Test ID

        Returns:
            List of statistical test results
        """
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")

        test = self.active_tests[test_id]

        try:
            # Get current metrics
            metrics = await self.analyze_test_performance(test_id)
            champion_metrics = metrics["champion"]
            challenger_metrics = metrics["challenger"]

            # Perform statistical tests
            test_results = []

            for metric_name in test.configuration.performance_metrics:
                if (
                    metric_name in champion_metrics.performance_metrics
                    and metric_name in challenger_metrics.performance_metrics
                ):
                    result = await self.statistical_analyzer.compare_metrics(
                        metric_name,
                        champion_metrics,
                        challenger_metrics,
                        test.configuration.significance_level,
                    )
                    test_results.append(result)

            return test_results

        except Exception as e:
            logger.error(f"Failed to check statistical significance for {test_id}: {e}")
            raise

    async def evaluate_test_completion(self, test_id: UUID) -> ABTestResult | None:
        """Evaluate if test should be completed and generate result.

        Args:
            test_id: Test ID

        Returns:
            Test result if test should be completed, None otherwise
        """
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")

        test = self.active_tests[test_id]

        try:
            # Check completion criteria
            should_complete, reason = await self._should_complete_test(test)

            if not should_complete:
                return None

            # Generate final result
            metrics = await self.analyze_test_performance(test_id)
            statistical_tests = await self.check_statistical_significance(test_id)

            # Determine conclusion
            conclusion = self._determine_test_conclusion(
                statistical_tests, test.configuration
            )

            # Calculate confidence
            confidence = self._calculate_conclusion_confidence(statistical_tests)

            # Generate recommendation
            recommendation = self._generate_recommendation(
                conclusion, statistical_tests, metrics
            )

            # Risk assessment
            risk_assessment = await self._assess_deployment_risk(
                metrics["challenger"], test.configuration
            )

            result = ABTestResult(
                test_id=test_id,
                test_name=test.test_name,
                conclusion=conclusion,
                confidence=confidence,
                champion_metrics=metrics["champion"],
                challenger_metrics=metrics["challenger"],
                statistical_tests=statistical_tests,
                duration=datetime.utcnow() - test.started_at,
                total_samples=len(self.prediction_records.get(test_id, [])),
                recommendation=recommendation,
                risk_assessment=risk_assessment,
            )

            # Update test status
            test.status = ABTestStatus.COMPLETED
            test.completed_at = datetime.utcnow()
            test.completion_reason = reason
            await self._save_test(test)

            logger.info(f"Completed A/B test {test_id}: {conclusion.value}")
            return result

        except Exception as e:
            logger.error(f"Failed to evaluate test completion for {test_id}: {e}")
            raise

    async def get_test_status(self, test_id: UUID) -> dict[str, Any]:
        """Get comprehensive status of A/B test.

        Args:
            test_id: Test ID

        Returns:
            Test status information
        """
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")

        test = self.active_tests[test_id]
        records = self.prediction_records.get(test_id, [])

        try:
            status = {
                "test_id": str(test_id),
                "test_name": test.test_name,
                "status": test.status.value,
                "created_at": test.created_at.isoformat(),
                "started_at": test.started_at.isoformat() if test.started_at else None,
                "duration": (
                    str(datetime.utcnow() - test.started_at)
                    if test.started_at
                    else None
                ),
                "total_predictions": len(records),
                "required_sample_size": test.required_sample_size,
                "completion_percentage": min(
                    100, len(records) / test.required_sample_size * 100
                ),
                "traffic_split": {
                    "champion": test.champion_variant.traffic_allocation,
                    "challenger": test.challenger_variant.traffic_allocation,
                },
                "configuration": {
                    "significance_level": test.configuration.significance_level,
                    "minimum_effect_size": test.configuration.minimum_effect_size,
                    "maximum_duration_days": test.configuration.maximum_duration_days,
                    "early_stopping_enabled": test.configuration.early_stopping_enabled,
                },
            }

            # Add variant-specific stats
            champion_records = [
                r for r in records if r.variant_id == test.champion_variant.variant_id
            ]
            challenger_records = [
                r for r in records if r.variant_id == test.challenger_variant.variant_id
            ]

            status["variant_stats"] = {
                "champion": {
                    "predictions": len(champion_records),
                    "feedback_rate": sum(
                        1 for r in champion_records if r.feedback_received
                    )
                    / max(1, len(champion_records)),
                },
                "challenger": {
                    "predictions": len(challenger_records),
                    "feedback_rate": sum(
                        1 for r in challenger_records if r.feedback_received
                    )
                    / max(1, len(challenger_records)),
                },
            }

            # Add current metrics if available
            if len(records) > 100:  # Minimum for meaningful metrics
                try:
                    current_metrics = await self.analyze_test_performance(test_id)
                    status["current_performance"] = {
                        "champion": current_metrics["champion"].performance_metrics,
                        "challenger": current_metrics["challenger"].performance_metrics,
                    }
                except Exception as e:
                    logger.warning(f"Failed to get current metrics: {e}")

            return status

        except Exception as e:
            logger.error(f"Failed to get test status for {test_id}: {e}")
            raise

    # Private helper methods

    def _calculate_sample_size(self, config: TestConfiguration) -> int:
        """Calculate required sample size for test."""
        # Simplified sample size calculation
        # In practice, would use power analysis

        alpha = config.significance_level
        beta = 1 - config.power_threshold
        effect_size = config.minimum_effect_size

        # Cohen's formula for two-sample t-test
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(1 - beta)

        n = 2 * ((z_alpha + z_beta) / effect_size) ** 2

        return max(int(n), config.minimum_sample_size)

    async def _calculate_variant_metrics(
        self,
        variant: ModelVariant,
        records: list[PredictionRecord],
        config: TestConfiguration,
    ) -> ABTestMetrics:
        """Calculate metrics for a variant."""
        if not records:
            return ABTestMetrics(
                variant_id=variant.variant_id, variant_name=variant.variant_name
            )

        # Filter records with feedback
        feedback_records = [r for r in records if r.feedback_received]

        metrics = ABTestMetrics(
            variant_id=variant.variant_id,
            variant_name=variant.variant_name,
            total_predictions=len(records),
            processing_times=[r.processing_time_ms for r in records],
        )

        if feedback_records:
            # Calculate performance metrics
            y_true = [r.actual_label for r in feedback_records]
            y_pred = [r.prediction for r in feedback_records]

            try:
                if "accuracy" in config.performance_metrics:
                    metrics.performance_metrics["accuracy"] = accuracy_score(
                        y_true, y_pred
                    )

                if "precision" in config.performance_metrics:
                    metrics.performance_metrics["precision"] = precision_score(
                        y_true, y_pred, average="weighted", zero_division=0
                    )

                if "recall" in config.performance_metrics:
                    metrics.performance_metrics["recall"] = recall_score(
                        y_true, y_pred, average="weighted", zero_division=0
                    )

                if "f1_score" in config.performance_metrics:
                    metrics.performance_metrics["f1_score"] = f1_score(
                        y_true, y_pred, average="weighted", zero_division=0
                    )

                # Update correct predictions count
                metrics.correct_predictions = sum(
                    1 for t, p in zip(y_true, y_pred, strict=False) if t == p
                )

            except Exception as e:
                logger.warning(f"Failed to calculate performance metrics: {e}")

        return metrics

    async def _should_complete_test(self, test: ABTest) -> tuple[bool, str]:
        """Check if test should be completed."""
        records = self.prediction_records.get(test.test_id, [])

        # Check sample size
        if len(records) >= test.required_sample_size:
            return True, "sufficient_sample_size"

        # Check duration
        if test.started_at:
            duration = datetime.utcnow() - test.started_at
            max_duration = timedelta(days=test.configuration.maximum_duration_days)
            if duration >= max_duration:
                return True, "maximum_duration_reached"

        # Check early stopping (if enabled)
        if test.configuration.early_stopping_enabled and len(records) > 500:
            try:
                statistical_tests = await self.check_statistical_significance(
                    test.test_id
                )
                significant_tests = [t for t in statistical_tests if t.is_significant]

                if (
                    len(significant_tests) >= len(statistical_tests) * 0.8
                ):  # 80% of tests significant
                    return True, "early_stopping_significance"

            except Exception as e:
                logger.warning(f"Failed to check early stopping: {e}")

        return False, "continuing"

    def _determine_test_conclusion(
        self, statistical_tests: list[StatisticalTestResult], config: TestConfiguration
    ) -> TestConclusion:
        """Determine test conclusion from statistical tests."""
        if not statistical_tests:
            return TestConclusion.INCONCLUSIVE

        # Count wins for each variant
        champion_wins = 0
        challenger_wins = 0

        for test_result in statistical_tests:
            if test_result.is_significant:
                if test_result.challenger_value > test_result.champion_value:
                    challenger_wins += 1
                else:
                    champion_wins += 1

        # Determine conclusion
        total_significant = champion_wins + challenger_wins

        if total_significant == 0:
            return TestConclusion.NO_SIGNIFICANT_DIFFERENCE
        elif challenger_wins > champion_wins:
            return TestConclusion.CHALLENGER_WINS
        elif champion_wins > challenger_wins:
            return TestConclusion.CHAMPION_WINS
        else:
            return TestConclusion.NO_SIGNIFICANT_DIFFERENCE

    def _calculate_conclusion_confidence(
        self, statistical_tests: list[StatisticalTestResult]
    ) -> float:
        """Calculate confidence in test conclusion."""
        if not statistical_tests:
            return 0.0

        # Average power of significant tests
        significant_tests = [t for t in statistical_tests if t.is_significant]

        if not significant_tests:
            return 0.0

        average_power = np.mean([t.power for t in significant_tests])
        return float(average_power)

    def _generate_recommendation(
        self,
        conclusion: TestConclusion,
        statistical_tests: list[StatisticalTestResult],
        metrics: dict[str, ABTestMetrics],
    ) -> str:
        """Generate recommendation based on test results."""
        if conclusion == TestConclusion.CHALLENGER_WINS:
            return "Deploy challenger model to production"
        elif conclusion == TestConclusion.CHAMPION_WINS:
            return "Keep champion model in production"
        elif conclusion == TestConclusion.NO_SIGNIFICANT_DIFFERENCE:
            return "No significant difference detected; consider other factors"
        else:
            return "Test results inconclusive; consider extending test duration"

    async def _assess_deployment_risk(
        self, challenger_metrics: ABTestMetrics, config: TestConfiguration
    ) -> dict[str, float]:
        """Assess risk of deploying challenger model."""
        risk_factors = {
            "performance_risk": 0.0,
            "stability_risk": 0.0,
            "business_risk": 0.0,
            "operational_risk": 0.0,
        }

        # Performance risk based on confidence intervals
        if challenger_metrics.confidence_intervals:
            # Calculate risk based on lower bounds of confidence intervals
            performance_risk = 0.0
            for _metric, (
                lower,
                _upper,
            ) in challenger_metrics.confidence_intervals.items():
                if lower < 0.8:  # Threshold for acceptable performance
                    performance_risk += 0.2
            risk_factors["performance_risk"] = min(1.0, performance_risk)

        # Stability risk based on processing time variance
        if challenger_metrics.processing_times:
            time_variance = np.var(challenger_metrics.processing_times)
            risk_factors["stability_risk"] = min(1.0, time_variance / 1000)  # Normalize

        # Business risk (simplified)
        risk_factors["business_risk"] = 0.1  # Baseline business risk

        # Operational risk based on error rate
        risk_factors["operational_risk"] = challenger_metrics.error_rate

        return risk_factors

    async def _save_test(self, test: ABTest) -> None:
        """Save A/B test to storage."""
        test_file = self.storage_path / f"test_{test.test_id}.json"

        # Convert test to serializable format (simplified)
        test_data = {
            "test_id": str(test.test_id),
            "test_name": test.test_name,
            "description": test.description,
            "status": test.status.value,
            "created_at": test.created_at.isoformat(),
            "started_at": test.started_at.isoformat() if test.started_at else None,
            "completed_at": (
                test.completed_at.isoformat() if test.completed_at else None
            ),
            "required_sample_size": test.required_sample_size,
            "configuration": {
                "test_name": test.configuration.test_name,
                "traffic_split_strategy": test.configuration.traffic_split_strategy.value,
                "champion_traffic_percentage": test.configuration.champion_traffic_percentage,
                "challenger_traffic_percentage": test.configuration.challenger_traffic_percentage,
                "significance_level": test.configuration.significance_level,
                "minimum_effect_size": test.configuration.minimum_effect_size,
            },
        }

        with open(test_file, "w") as f:
            json.dump(test_data, f, indent=2)

    async def _load_test_from_file(self, test_file: Path) -> ABTest:
        """Load A/B test from file."""
        # Simplified loading - in practice would reconstruct full test
        with open(test_file) as f:
            data = json.load(f)

        # This is a placeholder - would need full reconstruction
        from dataclasses import dataclass

        @dataclass
        class ABTest:
            test_id: UUID
            test_name: str
            description: str = ""
            status: ABTestStatus = ABTestStatus.PLANNING
            created_at: datetime = field(default_factory=datetime.utcnow)
            started_at: datetime | None = None
            completed_at: datetime | None = None
            required_sample_size: int = 1000
            configuration: TestConfiguration | None = None
            champion_variant: ModelVariant | None = None
            challenger_variant: ModelVariant | None = None
            completion_reason: str | None = None
            error_message: str | None = None

        test = ABTest(
            test_id=UUID(data["test_id"]),
            test_name=data["test_name"],
            description=data["description"],
            status=ABTestStatus(data["status"]),
            required_sample_size=data["required_sample_size"],
        )

        return test

    async def _monitor_test(self, test_id: UUID) -> None:
        """Monitor test for early stopping conditions."""
        while test_id in self.active_tests:
            test = self.active_tests[test_id]

            if test.status != ABTestStatus.RUNNING:
                break

            try:
                # Check for completion
                result = await self.evaluate_test_completion(test_id)
                if result:
                    break

                # Check guardrails
                if test.configuration.guardrail_metrics:
                    await self.guardrail_monitor.check_guardrails(
                        test_id, test.configuration
                    )

                # Wait for next check
                await asyncio.sleep(
                    test.configuration.early_stopping_check_interval_hours * 3600
                )

            except Exception as e:
                logger.error(f"Error monitoring test {test_id}: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour before retrying


# Supporting classes


class TrafficRouter:
    """Routes traffic between model variants."""

    def __init__(self):
        self.test_configurations: dict[UUID, TestConfiguration] = {}

    async def configure_test(self, test: ABTest) -> None:
        """Configure traffic routing for test."""
        self.test_configurations[test.test_id] = test.configuration

    async def route_request(
        self, test: ABTest, request_metadata: dict[str, Any] | None = None
    ) -> ModelVariant:
        """Route request to appropriate variant."""
        config = test.configuration

        if config.traffic_split_strategy == TrafficSplitStrategy.RANDOM:
            # Random assignment
            if np.random.random() < config.champion_traffic_percentage:
                return test.champion_variant
            else:
                return test.challenger_variant

        # Other strategies would be implemented here
        # For now, default to random
        if np.random.random() < 0.5:
            return test.champion_variant
        else:
            return test.challenger_variant


class StatisticalAnalyzer:
    """Performs statistical analysis for A/B tests."""

    async def compare_metrics(
        self,
        metric_name: str,
        champion_metrics: ABTestMetrics,
        challenger_metrics: ABTestMetrics,
        significance_level: float,
    ) -> StatisticalTestResult:
        """Compare metric between champion and challenger."""
        champion_value = champion_metrics.performance_metrics.get(metric_name, 0.0)
        challenger_value = challenger_metrics.performance_metrics.get(metric_name, 0.0)

        # Simplified statistical test (would use actual data points in practice)
        # For now, use normal approximation

        # Calculate standard errors (simplified)
        champion_se = 0.01  # Would calculate from actual data
        challenger_se = 0.01

        # Two-sample z-test
        diff = challenger_value - champion_value
        se_diff = np.sqrt(champion_se**2 + challenger_se**2)

        if se_diff > 0:
            z_stat = diff / se_diff
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        else:
            z_stat = 0.0
            p_value = 1.0

        is_significant = p_value < significance_level
        effect_size = abs(diff)

        # Calculate confidence interval
        ci_margin = stats.norm.ppf(1 - significance_level / 2) * se_diff
        confidence_interval = (diff - ci_margin, diff + ci_margin)

        # Calculate power (simplified)
        power = 0.8 if is_significant else 0.5

        # Generate interpretation
        if is_significant:
            if challenger_value > champion_value:
                interpretation = (
                    f"Challenger significantly outperforms champion on {metric_name}"
                )
            else:
                interpretation = (
                    f"Champion significantly outperforms challenger on {metric_name}"
                )
        else:
            interpretation = f"No significant difference in {metric_name}"

        return StatisticalTestResult(
            test_name="two_sample_z_test",
            metric_name=metric_name,
            champion_value=champion_value,
            challenger_value=challenger_value,
            test_statistic=z_stat,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=confidence_interval,
            is_significant=is_significant,
            power=power,
            interpretation=interpretation,
        )


class GuardrailMonitor:
    """Monitors guardrail metrics during A/B tests."""

    async def check_guardrails(
        self, test_id: UUID, config: TestConfiguration
    ) -> dict[str, bool]:
        """Check guardrail metrics for test."""
        # Placeholder implementation
        return {"all_guardrails_passing": True}


# Placeholder for missing ABTest dataclass that would be defined elsewhere
@dataclass
class ABTest:
    """A/B test entity."""

    test_id: UUID = field(default_factory=uuid4)
    test_name: str = ""
    description: str = ""
    status: ABTestStatus = ABTestStatus.PLANNING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    required_sample_size: int = 1000
    configuration: TestConfiguration | None = None
    champion_variant: ModelVariant | None = None
    challenger_variant: ModelVariant | None = None
    completion_reason: str | None = None
    error_message: str | None = None
