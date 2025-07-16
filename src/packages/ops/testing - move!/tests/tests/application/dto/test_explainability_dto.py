"""
Comprehensive tests for explainability DTOs.

This module tests all explainability-related Data Transfer Objects to ensure proper validation,
serialization, and behavior across all use cases including explanation methods, bias detection,
trust metrics, and cohort explanations.
"""

import pytest
from pydantic import ValidationError

from monorepo.application.dto.explainability_dto import (
    BiasMetric,
    CohortExplanationRequestDTO,
    ExplanationMethod,
    ExplanationType,
    TrustMetric,
)


class TestExplanationMethod:
    """Test suite for ExplanationMethod enum."""

    def test_explanation_method_values(self):
        """Test ExplanationMethod enum values."""
        assert ExplanationMethod.SHAP.value == "shap"
        assert ExplanationMethod.LIME.value == "lime"
        assert ExplanationMethod.PERMUTATION.value == "permutation"
        assert ExplanationMethod.INTEGRATED_GRADIENTS.value == "integrated_gradients"
        assert ExplanationMethod.GRADCAM.value == "gradcam"
        assert ExplanationMethod.ANCHORS.value == "anchors"
        assert ExplanationMethod.CAPTUM.value == "captum"

    def test_explanation_method_completeness(self):
        """Test that all expected explanation methods are present."""
        expected_methods = {
            "shap",
            "lime",
            "permutation",
            "integrated_gradients",
            "gradcam",
            "anchors",
            "captum",
        }
        actual_methods = {method.value for method in ExplanationMethod}
        assert actual_methods == expected_methods

    def test_explanation_method_string_conversion(self):
        """Test that ExplanationMethod can be used as strings."""
        # Test that enum values work in string contexts
        method = ExplanationMethod.SHAP
        assert str(method) == "ExplanationMethod.SHAP"
        assert method.value == "shap"

        # Test comparison with strings
        assert method == "shap"
        assert ExplanationMethod.LIME == "lime"


class TestExplanationType:
    """Test suite for ExplanationType enum."""

    def test_explanation_type_values(self):
        """Test ExplanationType enum values."""
        assert ExplanationType.LOCAL.value == "local"
        assert ExplanationType.GLOBAL.value == "global"
        assert ExplanationType.COHORT.value == "cohort"
        assert ExplanationType.COMPARATIVE.value == "comparative"

    def test_explanation_type_completeness(self):
        """Test that all expected explanation types are present."""
        expected_types = {"local", "global", "cohort", "comparative"}
        actual_types = {exp_type.value for exp_type in ExplanationType}
        assert actual_types == expected_types

    def test_explanation_type_ordering(self):
        """Test explanation type logical ordering."""
        types = [
            ExplanationType.LOCAL,
            ExplanationType.GLOBAL,
            ExplanationType.COHORT,
            ExplanationType.COMPARATIVE,
        ]
        type_values = [exp_type.value for exp_type in types]
        expected_order = ["local", "global", "cohort", "comparative"]
        assert type_values == expected_order


class TestBiasMetric:
    """Test suite for BiasMetric enum."""

    def test_bias_metric_values(self):
        """Test BiasMetric enum values."""
        assert BiasMetric.DEMOGRAPHIC_PARITY.value == "demographic_parity"
        assert BiasMetric.EQUALIZED_ODDS.value == "equalized_odds"
        assert BiasMetric.STATISTICAL_PARITY.value == "statistical_parity"
        assert BiasMetric.INDIVIDUAL_FAIRNESS.value == "individual_fairness"
        assert BiasMetric.COUNTERFACTUAL_FAIRNESS.value == "counterfactual_fairness"
        assert BiasMetric.CALIBRATION.value == "calibration"

    def test_bias_metric_completeness(self):
        """Test that all expected bias metrics are present."""
        expected_metrics = {
            "demographic_parity",
            "equalized_odds",
            "statistical_parity",
            "individual_fairness",
            "counterfactual_fairness",
            "calibration",
        }
        actual_metrics = {metric.value for metric in BiasMetric}
        assert actual_metrics == expected_metrics

    def test_bias_metric_categories(self):
        """Test bias metrics by fairness categories."""
        # Group bias metrics
        group_fairness = {
            BiasMetric.DEMOGRAPHIC_PARITY,
            BiasMetric.EQUALIZED_ODDS,
            BiasMetric.STATISTICAL_PARITY,
        }

        # Individual bias metrics
        individual_fairness = {
            BiasMetric.INDIVIDUAL_FAIRNESS,
            BiasMetric.COUNTERFACTUAL_FAIRNESS,
        }

        # Calibration metrics
        calibration_metrics = {BiasMetric.CALIBRATION}

        all_metrics = group_fairness | individual_fairness | calibration_metrics
        assert len(all_metrics) == 6


class TestTrustMetric:
    """Test suite for TrustMetric enum."""

    def test_trust_metric_values(self):
        """Test TrustMetric enum values."""
        assert TrustMetric.CONSISTENCY.value == "consistency"
        assert TrustMetric.STABILITY.value == "stability"
        assert TrustMetric.FIDELITY.value == "fidelity"
        assert TrustMetric.ROBUSTNESS.value == "robustness"

    def test_trust_metric_completeness(self):
        """Test that all expected trust metrics are present."""
        expected_metrics = {"consistency", "stability", "fidelity", "robustness"}
        actual_metrics = {metric.value for metric in TrustMetric}
        assert actual_metrics == expected_metrics

    def test_trust_metric_characteristics(self):
        """Test trust metrics represent different aspects of model trust."""
        # Temporal stability metrics
        temporal_metrics = {TrustMetric.CONSISTENCY, TrustMetric.STABILITY}

        # Explanation quality metrics
        quality_metrics = {TrustMetric.FIDELITY, TrustMetric.ROBUSTNESS}

        all_metrics = temporal_metrics | quality_metrics
        assert len(all_metrics) == 4


class TestCohortExplanationRequestDTO:
    """Test suite for CohortExplanationRequestDTO."""

    def test_basic_creation(self):
        """Test basic cohort explanation request creation."""
        request = CohortExplanationRequestDTO(
            detector_id="detector_123", dataset_id="dataset_456"
        )

        assert request.detector_id == "detector_123"
        assert request.dataset_id == "dataset_456"
        assert request.cohort_indices is None  # Default
        assert request.cohort_definitions is None  # Default
        assert request.explanation_method == "shap"  # Default
        assert request.explanation_methods is None  # Default
        assert request.max_features == 10  # Default
        assert request.comparison_baseline is None  # Default
        assert request.include_statistics is True  # Default

    def test_creation_with_cohort_indices(self):
        """Test creation with specific cohort indices."""
        indices = [1, 5, 10, 15, 20]

        request = CohortExplanationRequestDTO(
            detector_id="detector_789",
            dataset_id="dataset_123",
            cohort_indices=indices,
            explanation_method="lime",
            max_features=15,
        )

        assert request.cohort_indices == indices
        assert request.explanation_method == "lime"
        assert request.max_features == 15

    def test_creation_with_cohort_definitions(self):
        """Test creation with cohort definitions."""
        definitions = [
            {"age": {">=": 30, "<=": 60}},
            {"gender": {"==": "female"}},
            {"income": {">": 50000}},
        ]

        request = CohortExplanationRequestDTO(
            detector_id="detector_def",
            dataset_id="dataset_def",
            cohort_definitions=definitions,
            explanation_method="permutation",
        )

        assert request.cohort_definitions == definitions
        assert request.explanation_method == "permutation"

    def test_creation_with_multiple_explanation_methods(self):
        """Test creation with multiple explanation methods."""
        methods = [
            ExplanationMethod.SHAP,
            ExplanationMethod.LIME,
            ExplanationMethod.PERMUTATION,
        ]

        request = CohortExplanationRequestDTO(
            detector_id="detector_multi",
            dataset_id="dataset_multi",
            explanation_methods=methods,
            include_statistics=False,
        )

        assert request.explanation_methods == methods
        assert request.include_statistics is False

    def test_max_features_validation(self):
        """Test max_features validation bounds."""
        # Test invalid values
        with pytest.raises(ValidationError):
            CohortExplanationRequestDTO(
                detector_id="test",
                dataset_id="test",
                max_features=0,  # Below minimum
            )

        with pytest.raises(ValidationError):
            CohortExplanationRequestDTO(
                detector_id="test",
                dataset_id="test",
                max_features=51,  # Above maximum
            )

        # Test valid boundary values
        for max_features in [1, 25, 50]:
            request = CohortExplanationRequestDTO(
                detector_id="test", dataset_id="test", max_features=max_features
            )
            assert request.max_features == max_features

    def test_creation_with_comparison_baseline(self):
        """Test creation with comparison baseline."""
        request = CohortExplanationRequestDTO(
            detector_id="detector_compare",
            dataset_id="dataset_compare",
            comparison_baseline="global_average",
            explanation_method="integrated_gradients",
        )

        assert request.comparison_baseline == "global_average"
        assert request.explanation_method == "integrated_gradients"

    def test_complete_creation(self):
        """Test creation with all fields specified."""
        indices = [0, 1, 2, 3, 4]
        definitions = [{"category": {"in": ["A", "B", "C"]}}]
        methods = [ExplanationMethod.SHAP, ExplanationMethod.ANCHORS]

        request = CohortExplanationRequestDTO(
            detector_id="detector_complete",
            dataset_id="dataset_complete",
            cohort_indices=indices,
            cohort_definitions=definitions,
            explanation_method="captum",
            explanation_methods=methods,
            max_features=20,
            comparison_baseline="median",
            include_statistics=True,
        )

        assert request.detector_id == "detector_complete"
        assert request.dataset_id == "dataset_complete"
        assert request.cohort_indices == indices
        assert request.cohort_definitions == definitions
        assert request.explanation_method == "captum"
        assert request.explanation_methods == methods
        assert request.max_features == 20
        assert request.comparison_baseline == "median"
        assert request.include_statistics is True

    def test_empty_cohort_indices(self):
        """Test with empty cohort indices list."""
        request = CohortExplanationRequestDTO(
            detector_id="detector_empty", dataset_id="dataset_empty", cohort_indices=[]
        )

        assert request.cohort_indices == []

    def test_empty_cohort_definitions(self):
        """Test with empty cohort definitions list."""
        request = CohortExplanationRequestDTO(
            detector_id="detector_empty_def",
            dataset_id="dataset_empty_def",
            cohort_definitions=[],
        )

        assert request.cohort_definitions == []

    def test_empty_explanation_methods(self):
        """Test with empty explanation methods list."""
        request = CohortExplanationRequestDTO(
            detector_id="detector_empty_methods",
            dataset_id="dataset_empty_methods",
            explanation_methods=[],
        )

        assert request.explanation_methods == []

    def test_mixed_cohort_specification(self):
        """Test with both indices and definitions specified."""
        indices = [1, 2, 3]
        definitions = [{"status": {"==": "active"}}]

        request = CohortExplanationRequestDTO(
            detector_id="detector_mixed",
            dataset_id="dataset_mixed",
            cohort_indices=indices,
            cohort_definitions=definitions,
        )

        # Both should be preserved
        assert request.cohort_indices == indices
        assert request.cohort_definitions == definitions

    def test_large_cohort_indices(self):
        """Test with large cohort indices."""
        large_indices = list(range(1000))

        request = CohortExplanationRequestDTO(
            detector_id="detector_large",
            dataset_id="dataset_large",
            cohort_indices=large_indices,
            max_features=50,
        )

        assert len(request.cohort_indices) == 1000
        assert request.cohort_indices[-1] == 999

    def test_complex_cohort_definitions(self):
        """Test with complex cohort definitions."""
        complex_definitions = [
            {
                "age": {">=": 25, "<=": 65},
                "income": {">": 30000},
                "education": {"in": ["bachelor", "master", "phd"]},
            },
            {"location": {"startswith": "US"}, "experience": {">=": 5}},
        ]

        request = CohortExplanationRequestDTO(
            detector_id="detector_complex",
            dataset_id="dataset_complex",
            cohort_definitions=complex_definitions,
            explanation_method="gradcam",
        )

        assert request.cohort_definitions == complex_definitions
        assert len(request.cohort_definitions) == 2

    def test_different_baselines(self):
        """Test with different comparison baselines."""
        baselines = ["global_mean", "global_median", "zero", "random", "uniform"]

        for baseline in baselines:
            request = CohortExplanationRequestDTO(
                detector_id="detector_baseline",
                dataset_id="dataset_baseline",
                comparison_baseline=baseline,
            )
            assert request.comparison_baseline == baseline

    def test_all_explanation_methods(self):
        """Test with all available explanation methods."""
        all_methods = [
            ExplanationMethod.SHAP,
            ExplanationMethod.LIME,
            ExplanationMethod.PERMUTATION,
            ExplanationMethod.INTEGRATED_GRADIENTS,
            ExplanationMethod.GRADCAM,
            ExplanationMethod.ANCHORS,
            ExplanationMethod.CAPTUM,
        ]

        request = CohortExplanationRequestDTO(
            detector_id="detector_all_methods",
            dataset_id="dataset_all_methods",
            explanation_methods=all_methods,
        )

        assert len(request.explanation_methods) == 7
        assert all(method in request.explanation_methods for method in all_methods)

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            CohortExplanationRequestDTO(
                detector_id="test",
                dataset_id="test",
                extra_field="not_allowed",  # type: ignore
            )


class TestExplainabilityDTOIntegration:
    """Integration tests for explainability DTOs."""

    def test_comprehensive_cohort_explanation_workflow(self):
        """Test complete cohort explanation workflow."""
        # Step 1: Define cohorts using both indices and definitions
        high_risk_indices = [5, 15, 25, 35, 45]
        demographic_definitions = [{"age": {">=": 50}}, {"risk_score": {">=": 0.8}}]

        # Step 2: Create explanation request with multiple methods
        request = CohortExplanationRequestDTO(
            detector_id="fraud_detector_v2",
            dataset_id="financial_transactions_2024",
            cohort_indices=high_risk_indices,
            cohort_definitions=demographic_definitions,
            explanation_method="shap",  # Primary method
            explanation_methods=[
                ExplanationMethod.SHAP,
                ExplanationMethod.LIME,
                ExplanationMethod.PERMUTATION,
            ],
            max_features=25,
            comparison_baseline="global_median",
            include_statistics=True,
        )

        # Verify workflow configuration
        assert request.detector_id == "fraud_detector_v2"
        assert len(request.cohort_indices) == 5
        assert len(request.cohort_definitions) == 2
        assert len(request.explanation_methods) == 3
        assert request.max_features == 25
        assert request.include_statistics is True

    def test_bias_and_trust_metric_combinations(self):
        """Test different combinations of bias and trust metrics."""
        # Test fairness-focused analysis
        fairness_metrics = [
            BiasMetric.DEMOGRAPHIC_PARITY,
            BiasMetric.EQUALIZED_ODDS,
            BiasMetric.INDIVIDUAL_FAIRNESS,
        ]

        # Test trust-focused analysis
        trust_metrics = [
            TrustMetric.CONSISTENCY,
            TrustMetric.STABILITY,
            TrustMetric.FIDELITY,
            TrustMetric.ROBUSTNESS,
        ]

        # Verify metric categories
        assert len(fairness_metrics) == 3
        assert len(trust_metrics) == 4

        # Test comprehensive analysis combining both
        all_metrics = set(fairness_metrics + trust_metrics)
        assert len(all_metrics) == 7  # No duplicates

    def test_explanation_method_combinations(self):
        """Test different explanation method combinations."""
        # Model-agnostic methods
        model_agnostic = [
            ExplanationMethod.SHAP,
            ExplanationMethod.LIME,
            ExplanationMethod.PERMUTATION,
        ]

        # Deep learning methods
        deep_learning = [
            ExplanationMethod.INTEGRATED_GRADIENTS,
            ExplanationMethod.GRADCAM,
            ExplanationMethod.CAPTUM,
        ]

        # Rule-based methods
        rule_based = [ExplanationMethod.ANCHORS]

        # Test different scenarios
        scenarios = [
            ("tabular_data", model_agnostic + rule_based),
            ("image_data", deep_learning + [ExplanationMethod.LIME]),
            (
                "text_data",
                [
                    ExplanationMethod.SHAP,
                    ExplanationMethod.LIME,
                    ExplanationMethod.ANCHORS,
                ],
            ),
            ("comprehensive", model_agnostic + deep_learning + rule_based),
        ]

        for scenario_name, methods in scenarios:
            request = CohortExplanationRequestDTO(
                detector_id=f"detector_{scenario_name}",
                dataset_id=f"dataset_{scenario_name}",
                explanation_methods=methods,
            )

            assert len(request.explanation_methods) == len(methods)
            assert all(method in request.explanation_methods for method in methods)

    def test_cohort_definition_patterns(self):
        """Test various cohort definition patterns."""
        # Numerical range patterns
        numerical_cohorts = [
            {"age": {">=": 18, "<=": 65}},
            {"income": {">": 50000, "<=": 200000}},
            {"credit_score": {">=": 700}},
        ]

        # Categorical patterns
        categorical_cohorts = [
            {"gender": {"==": "female"}},
            {"education": {"in": ["bachelor", "master", "phd"]}},
            {"occupation": {"not_in": ["student", "retired"]}},
        ]

        # Complex combined patterns
        complex_cohorts = [
            {
                "age": {">=": 30, "<=": 50},
                "income": {">": 75000},
                "location": {"startswith": "US"},
            }
        ]

        # Test each pattern type
        pattern_tests = [
            ("numerical", numerical_cohorts),
            ("categorical", categorical_cohorts),
            ("complex", complex_cohorts),
        ]

        for pattern_name, cohort_defs in pattern_tests:
            request = CohortExplanationRequestDTO(
                detector_id=f"detector_{pattern_name}",
                dataset_id=f"dataset_{pattern_name}",
                cohort_definitions=cohort_defs,
            )

            assert request.cohort_definitions == cohort_defs
            assert len(request.cohort_definitions) == len(cohort_defs)

    def test_explanation_request_edge_cases(self):
        """Test explanation request edge cases."""
        # Edge case 1: Minimum max_features
        min_features_request = CohortExplanationRequestDTO(
            detector_id="detector_min", dataset_id="dataset_min", max_features=1
        )
        assert min_features_request.max_features == 1

        # Edge case 2: Maximum max_features
        max_features_request = CohortExplanationRequestDTO(
            detector_id="detector_max", dataset_id="dataset_max", max_features=50
        )
        assert max_features_request.max_features == 50

        # Edge case 3: Single cohort index
        single_index_request = CohortExplanationRequestDTO(
            detector_id="detector_single",
            dataset_id="dataset_single",
            cohort_indices=[42],
        )
        assert len(single_index_request.cohort_indices) == 1
        assert single_index_request.cohort_indices[0] == 42

        # Edge case 4: Single explanation method
        single_method_request = CohortExplanationRequestDTO(
            detector_id="detector_single_method",
            dataset_id="dataset_single_method",
            explanation_methods=[ExplanationMethod.SHAP],
        )
        assert len(single_method_request.explanation_methods) == 1
        assert single_method_request.explanation_methods[0] == ExplanationMethod.SHAP

    def test_explainability_validation_scenarios(self):
        """Test explainability validation scenarios."""
        # Valid scenarios that should not raise exceptions
        valid_scenarios = [
            # Minimal request
            {"detector_id": "det1", "dataset_id": "data1"},
            # Indices-based cohort
            {"detector_id": "det2", "dataset_id": "data2", "cohort_indices": [1, 2, 3]},
            # Definition-based cohort
            {
                "detector_id": "det3",
                "dataset_id": "data3",
                "cohort_definitions": [{"category": {"==": "A"}}],
            },
            # Multiple methods
            {
                "detector_id": "det4",
                "dataset_id": "data4",
                "explanation_methods": [ExplanationMethod.SHAP, ExplanationMethod.LIME],
            },
        ]

        for scenario in valid_scenarios:
            request = CohortExplanationRequestDTO(**scenario)
            assert request.detector_id.startswith("det")
            assert request.dataset_id.startswith("data")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
