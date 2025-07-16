"""
Tests for AnomalyExplanationDTO.

This module tests the AnomalyExplanationDTO class to ensure proper validation,
serialization, and behavior across all use cases.
"""

import pytest
from pydantic import ValidationError

from pynomaly.application.dto.explainability_dto import AnomalyExplanationDTO


class TestAnomalyExplanationDTO:
    """Test suite for AnomalyExplanationDTO."""

    def test_basic_creation(self):
        """Test basic AnomalyExplanationDTO creation."""
        dto = AnomalyExplanationDTO(
            sample_id=123,
            anomaly_score=0.85,
            explanation_confidence=0.9,
            explanation_method="shap",
        )

        assert dto.sample_id == 123
        assert dto.anomaly_score == 0.85
        assert dto.explanation_confidence == 0.9
        assert dto.explanation_method == "shap"
        assert dto.contributing_features == {}
        assert dto.feature_importances == {}
        assert dto.normal_range_deviations == {}
        assert dto.similar_normal_samples == []

    def test_complete_creation(self):
        """Test complete AnomalyExplanationDTO creation with all fields."""
        dto = AnomalyExplanationDTO(
            sample_id=456,
            anomaly_score=0.92,
            contributing_features={"feature1": 0.6, "feature2": -0.3},
            feature_importances={"feature1": 0.8, "feature2": 0.4},
            normal_range_deviations={"feature1": 2.5, "feature2": -1.8},
            similar_normal_samples=[1, 5, 10],
            explanation_confidence=0.95,
            explanation_method="lime",
        )

        assert dto.sample_id == 456
        assert dto.anomaly_score == 0.92
        assert dto.contributing_features == {"feature1": 0.6, "feature2": -0.3}
        assert dto.feature_importances == {"feature1": 0.8, "feature2": 0.4}
        assert dto.normal_range_deviations == {"feature1": 2.5, "feature2": -1.8}
        assert dto.similar_normal_samples == [1, 5, 10]
        assert dto.explanation_confidence == 0.95
        assert dto.explanation_method == "lime"

    def test_explanation_confidence_validation(self):
        """Test explanation confidence validation bounds."""
        # Test invalid values
        with pytest.raises(ValidationError):
            AnomalyExplanationDTO(
                sample_id=1,
                anomaly_score=0.5,
                explanation_confidence=-0.1,  # Below minimum
                explanation_method="shap",
            )

        with pytest.raises(ValidationError):
            AnomalyExplanationDTO(
                sample_id=1,
                anomaly_score=0.5,
                explanation_confidence=1.1,  # Above maximum
                explanation_method="shap",
            )

        # Test valid boundary values
        for confidence in [0.0, 0.5, 1.0]:
            dto = AnomalyExplanationDTO(
                sample_id=1,
                anomaly_score=0.5,
                explanation_confidence=confidence,
                explanation_method="shap",
            )
            assert dto.explanation_confidence == confidence

    def test_feature_contribution_patterns(self):
        """Test various feature contribution patterns."""
        # Test positive contributions
        dto_positive = AnomalyExplanationDTO(
            sample_id=1,
            anomaly_score=0.8,
            contributing_features={"temp": 0.5, "pressure": 0.3, "humidity": 0.2},
            explanation_confidence=0.9,
            explanation_method="permutation",
        )

        assert all(value > 0 for value in dto_positive.contributing_features.values())

        # Test mixed contributions
        dto_mixed = AnomalyExplanationDTO(
            sample_id=2,
            anomaly_score=0.7,
            contributing_features={"temp": 0.6, "pressure": -0.2, "humidity": 0.1},
            explanation_confidence=0.85,
            explanation_method="shap",
        )

        assert dto_mixed.contributing_features["temp"] > 0
        assert dto_mixed.contributing_features["pressure"] < 0
        assert dto_mixed.contributing_features["humidity"] > 0

    def test_normal_range_deviations_interpretation(self):
        """Test normal range deviations interpretation."""
        dto = AnomalyExplanationDTO(
            sample_id=3,
            anomaly_score=0.95,
            normal_range_deviations={
                "temperature": 3.2,  # Significantly above normal
                "humidity": -2.1,  # Significantly below normal
                "pressure": 0.1,  # Within normal range
            },
            explanation_confidence=0.92,
            explanation_method="lime",
        )

        assert (
            dto.normal_range_deviations["temperature"] > 3.0
        )  # High positive deviation
        assert dto.normal_range_deviations["humidity"] < -2.0  # High negative deviation
        assert abs(dto.normal_range_deviations["pressure"]) < 0.5  # Low deviation

    def test_similar_normal_samples_utility(self):
        """Test similar normal samples utility."""
        # Test with multiple similar samples
        dto_multiple = AnomalyExplanationDTO(
            sample_id=4,
            anomaly_score=0.6,
            similar_normal_samples=[10, 25, 67, 89, 123],
            explanation_confidence=0.8,
            explanation_method="permutation",
        )

        assert len(dto_multiple.similar_normal_samples) == 5
        assert all(
            isinstance(sample_id, int)
            for sample_id in dto_multiple.similar_normal_samples
        )

        # Test with no similar samples
        dto_no_similar = AnomalyExplanationDTO(
            sample_id=5,
            anomaly_score=0.99,  # Very anomalous, no similar normal samples
            similar_normal_samples=[],
            explanation_confidence=0.95,
            explanation_method="shap",
        )

        assert len(dto_no_similar.similar_normal_samples) == 0

    def test_multiple_explanation_methods(self):
        """Test creation with different explanation methods."""
        methods = [
            "shap",
            "lime",
            "permutation",
            "integrated_gradients",
            "gradcam",
            "anchors",
            "captum",
        ]

        for method in methods:
            dto = AnomalyExplanationDTO(
                sample_id=1,
                anomaly_score=0.8,
                explanation_confidence=0.9,
                explanation_method=method,
            )
            assert dto.explanation_method == method

    def test_empty_feature_dictionaries(self):
        """Test with empty feature dictionaries."""
        dto = AnomalyExplanationDTO(
            sample_id=6,
            anomaly_score=0.75,
            contributing_features={},
            feature_importances={},
            normal_range_deviations={},
            explanation_confidence=0.7,
            explanation_method="shap",
        )

        assert dto.contributing_features == {}
        assert dto.feature_importances == {}
        assert dto.normal_range_deviations == {}

    def test_large_feature_sets(self):
        """Test with large feature sets."""
        # Create large feature sets
        large_contributions = {f"feature_{i}": 0.1 * i for i in range(100)}
        large_importances = {f"feature_{i}": 0.05 * i for i in range(100)}
        large_deviations = {f"feature_{i}": 0.2 * i - 10 for i in range(100)}

        dto = AnomalyExplanationDTO(
            sample_id=7,
            anomaly_score=0.88,
            contributing_features=large_contributions,
            feature_importances=large_importances,
            normal_range_deviations=large_deviations,
            explanation_confidence=0.85,
            explanation_method="lime",
        )

        assert len(dto.contributing_features) == 100
        assert len(dto.feature_importances) == 100
        assert len(dto.normal_range_deviations) == 100

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            AnomalyExplanationDTO(
                sample_id=1,
                anomaly_score=0.8,
                explanation_confidence=0.9,
                explanation_method="shap",
                extra_field="not_allowed",  # type: ignore
            )


class TestExplainabilityIntegration:
    """Integration tests for explainability components."""

    def test_cohort_explanation_workflow(self):
        """Test cohort explanation workflow with AnomalyExplanationDTO."""
        # Create multiple anomaly explanations for a cohort
        explanations = []

        for i in range(5):
            explanation = AnomalyExplanationDTO(
                sample_id=i,
                anomaly_score=0.7 + (i * 0.05),
                contributing_features={
                    "temperature": 0.4 + (i * 0.1),
                    "pressure": 0.3 - (i * 0.05),
                    "humidity": 0.2 + (i * 0.02),
                },
                feature_importances={
                    "temperature": 0.6,
                    "pressure": 0.25,
                    "humidity": 0.15,
                },
                explanation_confidence=0.8 + (i * 0.02),
                explanation_method="shap",
            )
            explanations.append(explanation)

        # Verify cohort has consistent structure
        assert len(explanations) == 5
        assert all(exp.explanation_method == "shap" for exp in explanations)
        assert all("temperature" in exp.contributing_features for exp in explanations)

        # Verify progressive scoring
        scores = [exp.anomaly_score for exp in explanations]
        assert scores == sorted(scores)  # Should be in ascending order

        # Verify feature consistency
        for exp in explanations:
            assert set(exp.contributing_features.keys()) == {
                "temperature",
                "pressure",
                "humidity",
            }
            assert set(exp.feature_importances.keys()) == {
                "temperature",
                "pressure",
                "humidity",
            }
