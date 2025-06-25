"""Tests for advanced explainability service."""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from pynomaly.application.services.advanced_explainability_service import (
    AdvancedExplainabilityService,
    BiasAnalysisConfig,
    BiasAnalysisResult,
    ExplanationConfig,
    ExplanationReport,
    GlobalExplanation,
    LocalExplanation,
    TrustScoreConfig,
    TrustScoreResult,
)
from pynomaly.domain.entities import Dataset


class TestAdvancedExplainabilityService:
    """Test advanced explainability service."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service = AdvancedExplainabilityService()

        # Create test dataset
        np.random.seed(42)
        self.test_data = pd.DataFrame(
            {
                "feature_1": np.random.normal(0, 1, 1000),
                "feature_2": np.random.normal(0, 1, 1000),
                "feature_3": np.random.normal(0, 1, 1000),
                "protected_attr": np.random.choice(["A", "B"], 1000),
            }
        )

        self.test_dataset = Dataset(
            name="test_dataset",
            data=self.test_data,
            features=["feature_1", "feature_2", "feature_3", "protected_attr"],
        )

        # Create mock detector
        self.mock_detector = Mock()
        self.mock_detector.decision_function.return_value = np.random.random(1000)
        self.mock_detector.predict.return_value = (np.random.random(1000) > 0.5).astype(
            int
        )
        self.mock_detector.algorithm_name = "TestDetector"
        self.mock_detector.algorithm_params = {"test": True}
        self.mock_detector.is_trained = True

    def test_initialization(self):
        """Test service initialization."""
        assert self.service is not None
        assert isinstance(self.service.explanation_cache, dict)
        assert isinstance(self.service.explainer_cache, dict)

    def test_initialization_with_options(self):
        """Test service initialization with options."""
        service = AdvancedExplainabilityService(
            enable_shap=False,
            enable_lime=False,
            enable_permutation=True,
            cache_explanations=False,
        )

        assert service.enable_shap is False
        assert service.enable_lime is False
        assert service.enable_permutation is True
        assert service.cache_explanations is False

    @pytest.mark.asyncio
    async def test_generate_comprehensive_explanation(self):
        """Test comprehensive explanation generation."""
        config = ExplanationConfig(
            explanation_type="both",
            n_samples=5,
            feature_names=["feature_1", "feature_2", "feature_3"],
        )

        # Mock the internal methods
        with (
            patch.object(self.service, "_generate_local_explanations") as mock_local,
            patch.object(self.service, "_generate_global_explanation") as mock_global,
            patch.object(self.service, "_assess_trust_score") as mock_trust,
        ):
            # Set up mock returns
            mock_local.return_value = [
                LocalExplanation(
                    sample_id="sample_0",
                    prediction=0.5,
                    confidence=0.8,
                    feature_contributions={"feature_1": 0.1, "feature_2": -0.05},
                    explanation_method="mock",
                )
            ]

            mock_global.return_value = GlobalExplanation(
                feature_importance={
                    "feature_1": 0.5,
                    "feature_2": 0.3,
                    "feature_3": 0.2,
                },
                feature_interactions={},
                model_summary={"n_features": 3},
                explanation_method="mock",
                coverage=0.95,
                reliability=0.85,
            )

            mock_trust.return_value = TrustScoreResult(
                overall_trust_score=0.8,
                consistency_score=0.7,
                stability_score=0.8,
                fidelity_score=0.9,
                trust_factors={"consistency": 0.7, "stability": 0.8, "fidelity": 0.9},
                risk_assessment="medium",
            )

            # Generate explanation
            report = await self.service.generate_comprehensive_explanation(
                detector=self.mock_detector, dataset=self.test_dataset, config=config
            )

            # Verify report structure
            assert isinstance(report, ExplanationReport)
            assert len(report.local_explanations) == 1
            assert report.global_explanation is not None
            assert report.trust_assessment is not None
            assert isinstance(report.recommendations, list)

    @pytest.mark.asyncio
    async def test_generate_local_explanations(self):
        """Test local explanation generation."""
        X = self.test_data[["feature_1", "feature_2", "feature_3"]].values
        feature_names = ["feature_1", "feature_2", "feature_3"]
        config = ExplanationConfig(n_samples=3)

        # Mock SHAP computation
        with patch.object(self.service, "_compute_shap_values") as mock_shap:
            mock_shap.return_value = {
                "feature_1": 0.1,
                "feature_2": -0.05,
                "feature_3": 0.02,
            }

            explanations = await self.service._generate_local_explanations(
                self.mock_detector, X, feature_names, config
            )

            assert isinstance(explanations, list)
            assert len(explanations) <= config.n_samples

            if explanations:
                exp = explanations[0]
                assert isinstance(exp, LocalExplanation)
                assert exp.feature_contributions is not None
                assert exp.prediction is not None
                assert exp.confidence is not None

    @pytest.mark.asyncio
    async def test_generate_global_explanation(self):
        """Test global explanation generation."""
        X = self.test_data[["feature_1", "feature_2", "feature_3"]].values
        feature_names = ["feature_1", "feature_2", "feature_3"]
        config = ExplanationConfig()

        # Mock permutation importance
        with patch.object(self.service, "_compute_permutation_importance") as mock_perm:
            mock_perm.return_value = {
                "feature_1": 0.5,
                "feature_2": 0.3,
                "feature_3": 0.2,
            }

            explanation = await self.service._generate_global_explanation(
                self.mock_detector, X, feature_names, config
            )

            assert isinstance(explanation, GlobalExplanation)
            assert explanation.feature_importance is not None
            assert explanation.explanation_method is not None
            assert explanation.coverage is not None
            assert explanation.reliability is not None

    @pytest.mark.asyncio
    async def test_analyze_bias(self):
        """Test bias analysis."""
        config = BiasAnalysisConfig(
            protected_attributes=["protected_attr"],
            fairness_metrics=["demographic_parity"],
            min_group_size=10,
        )

        results = await self.service.analyze_bias(
            detector=self.mock_detector, dataset=self.test_dataset, config=config
        )

        assert isinstance(results, list)

        if results:
            result = results[0]
            assert isinstance(result, BiasAnalysisResult)
            assert result.protected_attribute == "protected_attr"
            assert isinstance(result.bias_detected, bool)
            assert result.severity in ["none", "low", "medium", "high"]

    @pytest.mark.asyncio
    async def test_analyze_attribute_bias(self):
        """Test bias analysis for specific attribute."""
        attribute_values = np.array(["A"] * 500 + ["B"] * 500)
        predictions = np.random.randint(0, 2, 1000)
        scores = np.random.random(1000)

        config = BiasAnalysisConfig(
            protected_attributes=["test_attr"], min_group_size=10
        )

        result = await self.service._analyze_attribute_bias(
            "test_attr", attribute_values, predictions, scores, config
        )

        assert isinstance(result, BiasAnalysisResult)
        assert result.protected_attribute == "test_attr"
        assert isinstance(result.fairness_metrics, dict)
        assert isinstance(result.group_statistics, dict)
        assert isinstance(result.bias_detected, bool)

    @pytest.mark.asyncio
    async def test_assess_trust_score(self):
        """Test trust score assessment."""
        X = self.test_data[["feature_1", "feature_2", "feature_3"]].values
        predictions = self.mock_detector.decision_function(X)
        config = TrustScoreConfig()

        trust_result = await self.service._assess_trust_score(
            self.mock_detector, X, predictions, config
        )

        assert isinstance(trust_result, TrustScoreResult)
        assert 0.0 <= trust_result.overall_trust_score <= 1.0
        assert 0.0 <= trust_result.consistency_score <= 1.0
        assert 0.0 <= trust_result.stability_score <= 1.0
        assert 0.0 <= trust_result.fidelity_score <= 1.0
        assert trust_result.risk_assessment in ["low", "medium", "high"]

    @pytest.mark.asyncio
    async def test_compute_shap_values_unavailable(self):
        """Test SHAP computation when unavailable."""
        service = AdvancedExplainabilityService(enable_shap=False)

        result = await service._compute_shap_values(
            self.mock_detector,
            np.random.random((1, 3)),
            np.random.random((100, 3)),
            ["f1", "f2", "f3"],
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_compute_lime_values_unavailable(self):
        """Test LIME computation when unavailable."""
        service = AdvancedExplainabilityService(enable_lime=False)

        result = await service._compute_lime_values(
            self.mock_detector,
            np.random.random((1, 3)),
            np.random.random((100, 3)),
            ["f1", "f2", "f3"],
        )

        assert result is None

    def test_compute_gradient_explanation(self):
        """Test gradient-based explanation computation."""
        sample = np.random.random((1, 3))
        feature_names = ["f1", "f2", "f3"]

        gradients = self.service._compute_gradient_explanation(
            self.mock_detector, sample, feature_names
        )

        assert isinstance(gradients, dict)
        assert len(gradients) == len(feature_names)
        assert all(isinstance(v, float) for v in gradients.values())

    @pytest.mark.asyncio
    async def test_compute_permutation_importance_unavailable(self):
        """Test permutation importance when unavailable."""
        service = AdvancedExplainabilityService(enable_permutation=False)

        result = await service._compute_permutation_importance(
            self.mock_detector, np.random.random((100, 3)), ["f1", "f2", "f3"]
        )

        assert result is None

    def test_compute_variance_importance(self):
        """Test variance-based importance computation."""
        X = np.random.random((100, 3))
        feature_names = ["f1", "f2", "f3"]

        importance = self.service._compute_variance_importance(X, feature_names)

        assert isinstance(importance, dict)
        assert len(importance) == len(feature_names)
        assert all(isinstance(v, float) for v in importance.values())
        assert abs(sum(importance.values()) - 1.0) < 1e-6  # Should sum to 1

    def test_compute_feature_interactions(self):
        """Test feature interaction computation."""
        X = np.random.random((100, 3))
        feature_names = ["f1", "f2", "f3"]

        interactions = self.service._compute_feature_interactions(X, feature_names)

        assert isinstance(interactions, dict)
        # Should have interactions for pairs of features
        assert len(interactions) <= len(feature_names) * (len(feature_names) - 1) // 2

    @pytest.mark.asyncio
    async def test_assess_consistency(self):
        """Test consistency assessment."""
        X = np.random.random((100, 3))

        consistency = await self.service._assess_consistency(self.mock_detector, X)

        assert isinstance(consistency, float)
        assert 0.0 <= consistency <= 1.0

    @pytest.mark.asyncio
    async def test_assess_stability(self):
        """Test stability assessment."""
        X = np.random.random((50, 3))
        config = TrustScoreConfig(n_perturbations=10)

        stability = await self.service._assess_stability(self.mock_detector, X, config)

        assert isinstance(stability, float)
        assert 0.0 <= stability <= 1.0

    @pytest.mark.asyncio
    async def test_assess_fidelity(self):
        """Test fidelity assessment."""
        X = np.random.random((50, 3))

        fidelity = await self.service._assess_fidelity(self.mock_detector, X)

        assert isinstance(fidelity, float)
        assert 0.0 <= fidelity <= 1.0

    def test_select_representative_samples(self):
        """Test representative sample selection."""
        X = np.random.random((200, 3))
        n_samples = 10

        # Test with sklearn available
        with patch("sklearn.cluster.KMeans") as mock_kmeans:
            mock_kmeans.return_value.fit_predict.return_value = np.random.randint(
                0, n_samples, 200
            )
            mock_kmeans.return_value.cluster_centers_ = np.random.random((n_samples, 3))

            indices = self.service._select_representative_samples(X, n_samples)

            assert isinstance(indices, list)
            assert len(indices) <= n_samples
            assert all(isinstance(idx, int | np.integer) for idx in indices)

    def test_select_representative_samples_fallback(self):
        """Test representative sample selection fallback."""
        X = np.random.random((200, 3))
        n_samples = 10

        # Test without sklearn clustering
        with patch("sklearn.cluster.KMeans", side_effect=ImportError()):
            indices = self.service._select_representative_samples(X, n_samples)

            assert isinstance(indices, list)
            assert len(indices) == n_samples

    def test_calculate_prediction_confidence(self):
        """Test prediction confidence calculation."""
        sample = np.random.random((1, 3))
        X = np.random.random((100, 3))

        confidence = self.service._calculate_prediction_confidence(
            self.mock_detector, sample, X
        )

        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    def test_get_model_info(self):
        """Test model info extraction."""
        info = self.service._get_model_info(self.mock_detector)

        assert isinstance(info, dict)
        assert "algorithm" in info
        assert "parameters" in info
        assert "is_trained" in info

    def test_get_dataset_summary(self):
        """Test dataset summary extraction."""
        summary = self.service._get_dataset_summary(self.test_dataset)

        assert isinstance(summary, dict)
        assert "name" in summary
        assert "n_samples" in summary
        assert "n_features" in summary
        assert "features" in summary

    def test_create_fallback_global_explanation(self):
        """Test fallback global explanation creation."""
        feature_names = ["f1", "f2", "f3"]

        explanation = self.service._create_fallback_global_explanation(feature_names)

        assert isinstance(explanation, GlobalExplanation)
        assert explanation.explanation_method == "fallback"
        assert len(explanation.feature_importance) == len(feature_names)
        assert explanation.coverage == 0.5
        assert explanation.reliability == 0.3

    def test_generate_recommendations(self):
        """Test recommendation generation."""
        trust_assessment = TrustScoreResult(
            overall_trust_score=0.5,
            consistency_score=0.6,
            stability_score=0.4,
            fidelity_score=0.8,
            trust_factors={},
            risk_assessment="high",
        )

        recommendations = self.service._generate_recommendations(trust_assessment)

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert all(isinstance(rec, str) for rec in recommendations)

    def test_get_service_info(self):
        """Test service information retrieval."""
        info = self.service.get_service_info()

        assert isinstance(info, dict)
        assert "shap_available" in info
        assert "lime_available" in info
        assert "sklearn_available" in info
        assert "shap_enabled" in info
        assert "lime_enabled" in info
        assert "permutation_enabled" in info
        assert "cache_enabled" in info
        assert "cached_explanations" in info
        assert "cached_explainers" in info


class TestExplanationConfig:
    """Test explanation configuration."""

    def test_default_config(self):
        """Test default configuration."""
        config = ExplanationConfig()

        assert config.explanation_type == "local"
        assert config.method == "shap"
        assert config.n_samples == 1000
        assert config.target_audience == "technical"
        assert config.include_confidence is True
        assert config.generate_plots is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = ExplanationConfig(
            explanation_type="global",
            method="lime",
            n_samples=50,
            feature_names=["f1", "f2"],
            target_audience="business",
            include_confidence=False,
            generate_plots=False,
        )

        assert config.explanation_type == "global"
        assert config.method == "lime"
        assert config.n_samples == 50
        assert config.feature_names == ["f1", "f2"]
        assert config.target_audience == "business"
        assert config.include_confidence is False
        assert config.generate_plots is False


class TestBiasAnalysisConfig:
    """Test bias analysis configuration."""

    def test_config_creation(self):
        """Test bias analysis config creation."""
        config = BiasAnalysisConfig(
            protected_attributes=["gender", "age"],
            fairness_metrics=["demographic_parity", "equalized_odds"],
            threshold=0.6,
            min_group_size=50,
        )

        assert config.protected_attributes == ["gender", "age"]
        assert config.fairness_metrics == ["demographic_parity", "equalized_odds"]
        assert config.threshold == 0.6
        assert config.min_group_size == 50


class TestTrustScoreConfig:
    """Test trust score configuration."""

    def test_config_creation(self):
        """Test trust score config creation."""
        config = TrustScoreConfig(
            consistency_checks=False,
            stability_analysis=True,
            fidelity_assessment=True,
            n_perturbations=200,
            perturbation_strength=0.2,
        )

        assert config.consistency_checks is False
        assert config.stability_analysis is True
        assert config.fidelity_assessment is True
        assert config.n_perturbations == 200
        assert config.perturbation_strength == 0.2


class TestExplanationModels:
    """Test explanation data models."""

    def test_local_explanation_creation(self):
        """Test local explanation model creation."""
        explanation = LocalExplanation(
            sample_id="test_sample",
            prediction=0.8,
            confidence=0.9,
            feature_contributions={"f1": 0.5, "f2": -0.3},
            explanation_method="shap",
            trust_score=0.85,
        )

        assert explanation.sample_id == "test_sample"
        assert explanation.prediction == 0.8
        assert explanation.confidence == 0.9
        assert explanation.feature_contributions == {"f1": 0.5, "f2": -0.3}
        assert explanation.explanation_method == "shap"
        assert explanation.trust_score == 0.85

    def test_global_explanation_creation(self):
        """Test global explanation model creation."""
        explanation = GlobalExplanation(
            feature_importance={"f1": 0.6, "f2": 0.4},
            feature_interactions={"f1_x_f2": 0.2},
            model_summary={"n_features": 2},
            explanation_method="permutation",
            coverage=0.95,
            reliability=0.88,
        )

        assert explanation.feature_importance == {"f1": 0.6, "f2": 0.4}
        assert explanation.feature_interactions == {"f1_x_f2": 0.2}
        assert explanation.model_summary == {"n_features": 2}
        assert explanation.explanation_method == "permutation"
        assert explanation.coverage == 0.95
        assert explanation.reliability == 0.88

    def test_bias_analysis_result_creation(self):
        """Test bias analysis result model creation."""
        result = BiasAnalysisResult(
            protected_attribute="gender",
            fairness_metrics={"demographic_parity": 0.85},
            group_statistics={"male": {"size": 500}, "female": {"size": 500}},
            bias_detected=False,
            severity="none",
            recommendations=["No bias detected"],
        )

        assert result.protected_attribute == "gender"
        assert result.fairness_metrics == {"demographic_parity": 0.85}
        assert result.bias_detected is False
        assert result.severity == "none"
        assert result.recommendations == ["No bias detected"]

    def test_trust_score_result_creation(self):
        """Test trust score result model creation."""
        result = TrustScoreResult(
            overall_trust_score=0.8,
            consistency_score=0.75,
            stability_score=0.85,
            fidelity_score=0.8,
            trust_factors={"consistency": 0.75, "stability": 0.85, "fidelity": 0.8},
            risk_assessment="medium",
        )

        assert result.overall_trust_score == 0.8
        assert result.consistency_score == 0.75
        assert result.stability_score == 0.85
        assert result.fidelity_score == 0.8
        assert result.risk_assessment == "medium"
