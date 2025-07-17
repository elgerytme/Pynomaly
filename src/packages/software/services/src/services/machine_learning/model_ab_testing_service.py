"""A/B testing service for processor experimentation."""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

import numpy as np
from scipy import stats

from monorepo.domain.entities.ab_test import (
    ABTest,
    ABTestConfiguration,
    ABTestMetrics,
    ABTestResult,
    ABTestStatus,
    ABTestSummary,
    SuccessMetric,
    TrafficSplit,
)
from monorepo.shared.protocols.repository_protocol import ModelRepositoryProtocol


class ModelABTestingService:
    """Service for managing A/B testing of models."""

    def __init__(
        self,
        processor_repository: ModelRepositoryProtocol,
        ab_test_repository: Any,  # ABTestRepositoryProtocol when implemented
        processor_serving_service: Any,  # ModelServingService when implemented
    ):
        """Initialize the A/B testing service.

        Args:
            processor_repository: Processor repository
            ab_test_repository: A/B test repository
            processor_serving_service: Processor serving service
        """
        self.processor_repository = processor_repository
        self.ab_test_repository = ab_test_repository
        self.processor_serving_service = processor_serving_service

    async def create_ab_test(
        self,
        name: str,
        control_processor_id: UUID,
        treatment_processor_id: UUID,
        traffic_split: TrafficSplit,
        duration: timedelta,
        success_measurements: list[SuccessMetric],
        created_by: str,
        description: str | None = None,
        min_sample_size: int = 100,
        confidence_level: float = 0.95,
        tags: list[str] | None = None,
    ) -> ABTest:
        """Create a new A/B test.

        Args:
            name: Test name
            control_processor_id: Control processor ID
            treatment_processor_id: Treatment processor ID
            traffic_split: Traffic allocation
            duration: Test duration
            success_measurements: Success measurements to evaluate
            created_by: User creating the test
            description: Test description
            min_sample_size: Minimum sample size
            confidence_level: Statistical confidence level
            tags: Test tags

        Returns:
            Created A/B test
        """
        # Validate models exist
        await self._validate_processors_exist([control_processor_id, treatment_processor_id])

        # Ensure models are different
        if control_processor_id == treatment_processor_id:
            raise ValueError("Control and treatment models must be different")

        # Validate at least one primary metric
        primary_measurements = [m for m in success_measurements if m.is_primary]
        if not primary_measurements:
            raise ValueError("At least one success metric must be marked as primary")

        # Create test configuration
        configuration = ABTestConfiguration(
            traffic_split=traffic_split,
            duration=duration,
            min_sample_size=min_sample_size,
            confidence_level=confidence_level,
            success_measurements=success_measurements,
        )

        # Create A/B test
        ab_test = ABTest(
            name=name,
            description=description,
            control_processor_id=control_processor_id,
            treatment_processor_id=treatment_processor_id,
            configuration=configuration,
            created_by=created_by,
            tags=tags or [],
        )

        # Store in repository
        stored_test = await self.ab_test_repository.create(ab_test)

        return stored_test

    async def start_ab_test(self, test_id: UUID) -> ABTest:
        """Start an A/B test.

        Args:
            test_id: Test ID to start

        Returns:
            Updated A/B test
        """
        ab_test = await self.ab_test_repository.get_by_id(test_id)
        if not ab_test:
            raise ValueError(f"A/B test {test_id} not found")

        # Validate models are ready for serving
        await self._validate_processors_ready(
            [ab_test.control_processor_id, ab_test.treatment_processor_id]
        )

        # Start the test
        ab_test.start_test()

        # Configure traffic routing in processor serving
        await self._configure_traffic_routing(ab_test)

        # Update in repository
        updated_test = await self.ab_test_repository.update(ab_test)

        return updated_test

    async def pause_ab_test(self, test_id: UUID) -> ABTest:
        """Pause an A/B test.

        Args:
            test_id: Test ID to pause

        Returns:
            Updated A/B test
        """
        ab_test = await self.ab_test_repository.get_by_id(test_id)
        if not ab_test:
            raise ValueError(f"A/B test {test_id} not found")

        ab_test.pause_test()

        # Remove traffic routing
        await self._remove_traffic_routing(ab_test)

        updated_test = await self.ab_test_repository.update(ab_test)

        return updated_test

    async def resume_ab_test(self, test_id: UUID) -> ABTest:
        """Resume a paused A/B test.

        Args:
            test_id: Test ID to resume

        Returns:
            Updated A/B test
        """
        ab_test = await self.ab_test_repository.get_by_id(test_id)
        if not ab_test:
            raise ValueError(f"A/B test {test_id} not found")

        ab_test.resume_test()

        # Reconfigure traffic routing
        await self._configure_traffic_routing(ab_test)

        updated_test = await self.ab_test_repository.update(ab_test)

        return updated_test

    async def complete_ab_test(
        self, test_id: UUID, force_completion: bool = False
    ) -> ABTest:
        """Complete an A/B test and analyze results.

        Args:
            test_id: Test ID to complete
            force_completion: Force completion even without sufficient data

        Returns:
            Completed A/B test with results
        """
        ab_test = await self.ab_test_repository.get_by_id(test_id)
        if not ab_test:
            raise ValueError(f"A/B test {test_id} not found")

        # Update measurements before completion
        await self.update_ab_test_measurements(test_id)
        ab_test = await self.ab_test_repository.get_by_id(test_id)

        # Analyze results
        result, conclusion = await self._analyze_ab_test_results(
            ab_test, force_completion
        )

        # Complete the test
        ab_test.complete_test(result, conclusion)

        # Remove traffic routing
        await self._remove_traffic_routing(ab_test)

        updated_test = await self.ab_test_repository.update(ab_test)

        return updated_test

    async def cancel_ab_test(self, test_id: UUID, reason: str | None = None) -> ABTest:
        """Cancel an A/B test.

        Args:
            test_id: Test ID to cancel
            reason: Cancellation reason

        Returns:
            Cancelled A/B test
        """
        ab_test = await self.ab_test_repository.get_by_id(test_id)
        if not ab_test:
            raise ValueError(f"A/B test {test_id} not found")

        ab_test.cancel_test(reason)

        # Remove traffic routing
        await self._remove_traffic_routing(ab_test)

        updated_test = await self.ab_test_repository.update(ab_test)

        return updated_test

    async def update_ab_test_measurements(self, test_id: UUID) -> ABTest:
        """Update A/B test measurements from processor serving data.

        Args:
            test_id: Test ID to update

        Returns:
            Updated A/B test
        """
        ab_test = await self.ab_test_repository.get_by_id(test_id)
        if not ab_test:
            raise ValueError(f"A/B test {test_id} not found")

        if not ab_test.is_active():
            return ab_test

        # Collect measurements from processor serving
        control_measurements = await self._collect_processor_measurements(
            ab_test.control_processor_id, ab_test.started_at
        )
        treatment_measurements = await self._collect_processor_measurements(
            ab_test.treatment_processor_id, ab_test.started_at
        )

        # Calculate statistical significance
        statistical_significance = {}
        p_values = {}
        confidence_intervals = {}
        effect_sizes = {}

        for metric in ab_test.configuration.success_measurements:
            if metric.name in control_measurements and metric.name in treatment_measurements:
                control_values = control_measurements[metric.name]
                treatment_values = treatment_measurements[metric.name]

                # Perform statistical test
                stat_result = self._perform_statistical_test(
                    control_values,
                    treatment_values,
                    ab_test.configuration.confidence_level,
                )

                statistical_significance[metric.name] = stat_result["significant"]
                p_values[metric.name] = stat_result["p_value"]
                confidence_intervals[metric.name] = stat_result["confidence_interval"]
                effect_sizes[metric.name] = stat_result["effect_size"]

        # Update measurements
        updated_measurements = ABTestMetrics(
            control_sample_size=len(control_measurements.get("sample_ids", [])),
            treatment_sample_size=len(treatment_measurements.get("sample_ids", [])),
            control_measurements={
                k: np.mean(v) for k, v in control_measurements.items() if k != "sample_ids"
            },
            treatment_measurements={
                k: np.mean(v) for k, v in treatment_measurements.items() if k != "sample_ids"
            },
            statistical_significance=statistical_significance,
            p_values=p_values,
            confidence_intervals=confidence_intervals,
            effect_sizes=effect_sizes,
        )

        ab_test.update_measurements(updated_measurements)

        # Check for early stopping
        if ab_test.configuration.early_stopping_enabled:
            should_stop = await self._check_early_stopping(ab_test)
            if should_stop:
                return await self.complete_ab_test(test_id)

        updated_test = await self.ab_test_repository.update(ab_test)

        return updated_test

    async def get_ab_test_summary(self, test_id: UUID) -> ABTestSummary:
        """Get A/B test summary.

        Args:
            test_id: Test ID

        Returns:
            A/B test summary
        """
        ab_test = await self.ab_test_repository.get_by_id(test_id)
        if not ab_test:
            raise ValueError(f"A/B test {test_id} not found")

        # Calculate primary metric improvement
        primary_metric_improvement = None
        primary_measurements = [
            m for m in ab_test.configuration.success_measurements if m.is_primary
        ]

        if (
            primary_measurements
            and primary_measurements[0].name in ab_test.current_measurements.control_measurements
        ):
            metric_name = primary_measurements[0].name
            control_value = ab_test.current_measurements.control_measurements[metric_name]
            treatment_value = ab_test.current_measurements.treatment_measurements[metric_name]

            if control_value > 0:
                primary_metric_improvement = (
                    (treatment_value - control_value) / control_value
                ) * 100

        # Check statistical significance
        is_significant = False
        if primary_measurements:
            metric_name = primary_measurements[0].name
            is_significant = ab_test.current_measurements.statistical_significance.get(
                metric_name, False
            )

        return ABTestSummary(
            test_id=ab_test.id,
            name=ab_test.name,
            status=ab_test.status,
            result=ab_test.result,
            control_processor_id=ab_test.control_processor_id,
            treatment_processor_id=ab_test.treatment_processor_id,
            started_at=ab_test.started_at,
            ended_at=ab_test.ended_at,
            sample_size=(
                ab_test.current_measurements.control_sample_size
                + ab_test.current_measurements.treatment_sample_size
            ),
            primary_metric_improvement=primary_metric_improvement,
            statistical_significance=is_significant,
            winning_processor_id=ab_test.get_winning_processor_id(),
        )

    async def list_ab_tests(
        self,
        processor_id: UUID | None = None,
        status: ABTestStatus | None = None,
        created_by: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ABTestSummary]:
        """List A/B tests with filters.

        Args:
            processor_id: Filter by processor ID
            status: Filter by status
            created_by: Filter by creator
            limit: Maximum results
            offset: Result offset

        Returns:
            List of A/B test summaries
        """
        ab_tests = await self.ab_test_repository.list_tests(
            processor_id=processor_id,
            status=status,
            created_by=created_by,
            limit=limit,
            offset=offset,
        )

        summaries = []
        for ab_test in ab_tests:
            summary = await self.get_ab_test_summary(ab_test.id)
            summaries.append(summary)

        return summaries

    async def get_processor_ab_tests(self, processor_id: UUID) -> list[ABTestSummary]:
        """Get all A/B tests for a processor.

        Args:
            processor_id: Processor ID

        Returns:
            List of A/B test summaries
        """
        return await self.list_ab_tests(processor_id=processor_id)

    async def route_prediction_request(
        self, request_data: dict[str, Any], user_id: str | None = None
    ) -> tuple[UUID, dict[str, Any]]:
        """Route prediction request based on active A/B tests.

        Args:
            request_data: Prediction request data
            user_id: User ID for consistent routing

        Returns:
            Tuple of (selected_processor_id, prediction_result)
        """
        # Get active A/B tests
        active_tests = await self.ab_test_repository.get_active_tests()

        if not active_tests:
            # No active tests, use default routing
            return await self._default_processor_routing(request_data)

        # For simplicity, use the first active test
        # In production, this would handle multiple tests with priorities
        ab_test = active_tests[0]

        # Determine which processor to use
        selected_processor_id = self._select_processor_for_request(ab_test, user_id)

        # Make prediction
        prediction_result = await self.processor_serving_service.predict(
            selected_processor_id, request_data
        )

        # Log the request for measurements collection
        await self._log_ab_test_request(
            ab_test.id, selected_processor_id, request_data, prediction_result
        )

        return selected_processor_id, prediction_result

    def _select_model_for_request(
        self, ab_test: ABTest, user_id: str | None = None
    ) -> UUID:
        """Select processor for a request based on traffic split.

        Args:
            ab_test: A/B test configuration
            user_id: User ID for consistent routing

        Returns:
            Selected processor ID
        """
        # Use user ID for consistent routing, otherwise random
        if user_id:
            # Hash user ID to get consistent assignment
            hash_value = hash(user_id) % 100
        else:
            hash_value = random.randint(0, 99)

        # Apply traffic split
        if hash_value < ab_test.configuration.traffic_split.control_percentage:
            return ab_test.control_processor_id
        else:
            return ab_test.treatment_processor_id

    async def _analyze_ab_test_results(
        self, ab_test: ABTest, force_completion: bool = False
    ) -> tuple[ABTestResult, str]:
        """Analyze A/B test results and determine winner.

        Args:
            ab_test: A/B test to analyze
            force_completion: Force completion even without sufficient data

        Returns:
            Tuple of (result, conclusion)
        """
        measurements = ab_test.current_measurements

        # Check sample size
        if not force_completion and not ab_test.has_sufficient_sample_size():
            return (
                ABTestResult.INCONCLUSIVE,
                "Insufficient sample size for reliable results",
            )

        # Analyze primary measurements
        primary_measurements = [
            m for m in ab_test.configuration.success_measurements if m.is_primary
        ]

        if not primary_measurements:
            return ABTestResult.INCONCLUSIVE, "No primary measurements defined"

        primary_metric = primary_measurements[0]

        # Check if we have data for the primary metric
        if primary_metric.name not in measurements.statistical_significance:
            return (
                ABTestResult.INCONCLUSIVE,
                f"No data available for primary metric: {primary_metric.name}",
            )

        is_significant = measurements.statistical_significance[primary_metric.name]

        if not is_significant:
            return (
                ABTestResult.NO_SIGNIFICANT_DIFFERENCE,
                "No statistically significant difference detected",
            )

        # Determine winner based on metric improvement
        control_value = measurements.control_measurements.get(primary_metric.name, 0)
        treatment_value = measurements.treatment_measurements.get(primary_metric.name, 0)

        if treatment_value > control_value:
            improvement = ((treatment_value - control_value) / control_value) * 100
            conclusion = f"Treatment processor wins with {improvement:.2f}% improvement in {primary_metric.name}"
            return ABTestResult.TREATMENT_WINS, conclusion
        else:
            degradation = ((control_value - treatment_value) / control_value) * 100
            conclusion = f"Control processor wins. Treatment processor shows {degradation:.2f}% degradation in {primary_metric.name}"
            return ABTestResult.CONTROL_WINS, conclusion

    def _perform_statistical_test(
        self,
        control_values: list[float],
        treatment_values: list[float],
        confidence_level: float,
    ) -> dict[str, Any]:
        """Perform statistical test to compare control and treatment.

        Args:
            control_values: Control group values
            treatment_values: Treatment group values
            confidence_level: Confidence level for the test

        Returns:
            Statistical test results
        """
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(treatment_values, control_values)

        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(
            (
                (len(control_values) - 1) * np.var(control_values, ddof=1)
                + (len(treatment_values) - 1) * np.var(treatment_values, ddof=1)
            )
            / (len(control_values) + len(treatment_values) - 2)
        )

        effect_size = (np.mean(treatment_values) - np.mean(control_values)) / pooled_std

        # Calculate confidence interval
        alpha = 1 - confidence_level
        degrees_freedom = len(control_values) + len(treatment_values) - 2
        t_critical = stats.t.ppf(1 - alpha / 2, degrees_freedom)

        mean_diff = np.mean(treatment_values) - np.mean(control_values)
        std_error = pooled_std * np.sqrt(
            1 / len(control_values) + 1 / len(treatment_values)
        )
        margin_error = t_critical * std_error

        confidence_interval = (mean_diff - margin_error, mean_diff + margin_error)

        return {
            "significant": p_value < alpha,
            "p_value": p_value,
            "effect_size": effect_size,
            "confidence_interval": confidence_interval,
            "t_statistic": t_stat,
        }

    async def _check_early_stopping(self, ab_test: ABTest) -> bool:
        """Check if test should be stopped early.

        Args:
            ab_test: A/B test to check

        Returns:
            Whether to stop early
        """
        # Simple early stopping: stop if we have clear winner with high confidence
        primary_measurements = [
            m for m in ab_test.configuration.success_measurements if m.is_primary
        ]

        if not primary_measurements:
            return False

        metric_name = primary_measurements[0].name

        # Check if we have statistical significance and sufficient sample size
        if (
            metric_name in ab_test.current_measurements.statistical_significance
            and ab_test.current_measurements.statistical_significance[metric_name]
            and ab_test.has_sufficient_sample_size()
        ):
            # Check effect size
            if metric_name in ab_test.current_measurements.effect_sizes:
                effect_size = abs(ab_test.current_measurements.effect_sizes[metric_name])
                # Stop early if large effect size (> 0.8 is considered large)
                return effect_size > 0.8

        return False

    async def _validate_processors_exist(self, processor_ids: list[UUID]) -> None:
        """Validate that models exist."""
        for processor_id in processor_ids:
            processor = await self.processor_repository.get_by_id(processor_id)
            if not processor:
                raise ValueError(f"Processor {processor_id} does not exist")

    async def _validate_processors_ready(self, processor_ids: list[UUID]) -> None:
        """Validate that models are ready for serving."""
        # This would check with the processor serving service
        # For now, just validate they exist
        await self._validate_processors_exist(processor_ids)

    async def _configure_traffic_routing(self, ab_test: ABTest) -> None:
        """Configure traffic routing for A/B test."""
        # This would configure the processor serving service to route traffic
        pass

    async def _remove_traffic_routing(self, ab_test: ABTest) -> None:
        """Remove traffic routing for A/B test."""
        # This would remove routing configuration from processor serving service
        pass

    async def _collect_processor_measurements(
        self, processor_id: UUID, since: datetime | None = None
    ) -> dict[str, list[float]]:
        """Collect measurements for a processor."""
        # This would collect actual measurements from the serving service
        # For now, return dummy data
        return {
            "accuracy": [0.85, 0.87, 0.86, 0.88],
            "precision": [0.82, 0.84, 0.83, 0.85],
            "recall": [0.89, 0.88, 0.90, 0.87],
            "f1_score": [0.85, 0.86, 0.86, 0.86],
            "sample_ids": [1, 2, 3, 4],
        }

    async def _log_ab_test_request(
        self, test_id: UUID, processor_id: UUID, request_data: dict, prediction_result: dict
    ) -> None:
        """Log A/B test request for measurements collection."""
        # This would log the request for later analysis
        pass

    async def _default_processor_routing(self, request_data: dict) -> tuple[UUID, dict]:
        """Default processor routing when no A/B tests are active."""
        # This would use default processor selection logic
        # For now, return dummy response
        default_processor_id = UUID("00000000-0000-0000-0000-000000000000")
        return default_processor_id, {"prediction": "dummy"}
