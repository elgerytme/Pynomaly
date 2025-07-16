"""Tests for Explainability DTOs."""

from datetime import datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from monorepo.application.dto.explainability_dto import (
    BiasAnalysisConfigDTO,
    BiasAnalysisResultDTO,
    BiasMetric,
    ComprehensiveExplanationRequestDTO,
    ComprehensiveExplanationResponseDTO,
    ExplanationConfigDTO,
    ExplanationMethod,
    ExplanationType,
    FeatureContributionDTO,
    FeatureInteractionDTO,
    GlobalExplanationDTO,
    LocalExplanationDTO,
    ReportGenerationConfigDTO,
    TrustAssessmentConfigDTO,
    TrustAssessmentResultDTO,
    TrustMetric,
    TrustMetricResultDTO,
    UncertaintyQuantificationDTO,
    VisualizationDataDTO,
    VisualizationType,
    calculate_explanation_quality_score,
    create_explanation_config_from_legacy,
    create_visualization_config,
    merge_feature_contributions,
    validate_explanation_request,
)


class TestExplanationConfigDTO:
    """Test suite for ExplanationConfigDTO."""

    def test_valid_creation(self):
        """Test creating a valid explanation config DTO."""
        dto = ExplanationConfigDTO(
            method=ExplanationMethod.SHAP,
            explanation_type=ExplanationType.LOCAL,
            max_features=15,
            background_samples=500,
            n_permutations=200,
            feature_names=["feature1", "feature2", "feature3"],
            categorical_features=["feature2"],
            random_state=42,
            shap_explainer_type="tree",
            shap_check_additivity=True,
            lime_mode="tabular",
            lime_kernel_width=0.75,
            lime_num_samples=3000,
            compute_interactions=True,
            include_confidence_intervals=True,
            parallel_processing=True,
            cache_explanations=True,
        )

        assert dto.method == ExplanationMethod.SHAP
        assert dto.explanation_type == ExplanationType.LOCAL
        assert dto.max_features == 15
        assert dto.background_samples == 500
        assert dto.n_permutations == 200
        assert dto.feature_names == ["feature1", "feature2", "feature3"]
        assert dto.categorical_features == ["feature2"]
        assert dto.random_state == 42
        assert dto.shap_explainer_type == "tree"
        assert dto.shap_check_additivity is True
        assert dto.lime_mode == "tabular"
        assert dto.lime_kernel_width == 0.75
        assert dto.lime_num_samples == 3000
        assert dto.compute_interactions is True
        assert dto.include_confidence_intervals is True
        assert dto.parallel_processing is True
        assert dto.cache_explanations is True

    def test_default_values(self):
        """Test default values."""
        dto = ExplanationConfigDTO(
            method=ExplanationMethod.LIME, explanation_type=ExplanationType.GLOBAL
        )

        assert dto.method == ExplanationMethod.LIME
        assert dto.explanation_type == ExplanationType.GLOBAL
        assert dto.max_features == 10
        assert dto.background_samples == 100
        assert dto.n_permutations == 100
        assert dto.feature_names is None
        assert dto.categorical_features is None
        assert dto.random_state == 42
        assert dto.shap_explainer_type == "auto"
        assert dto.shap_check_additivity is False
        assert dto.lime_mode == "tabular"
        assert dto.lime_kernel_width is None
        assert dto.lime_num_samples == 5000
        assert dto.compute_interactions is False
        assert dto.include_confidence_intervals is False
        assert dto.parallel_processing is True
        assert dto.cache_explanations is True

    def test_max_features_validation(self):
        """Test max features validation."""
        # Valid range
        dto = ExplanationConfigDTO(
            method=ExplanationMethod.SHAP,
            explanation_type=ExplanationType.LOCAL,
            max_features=50,
        )
        assert dto.max_features == 50

        # Invalid: too small
        with pytest.raises(ValidationError):
            ExplanationConfigDTO(
                method=ExplanationMethod.SHAP,
                explanation_type=ExplanationType.LOCAL,
                max_features=0,
            )

        # Invalid: too large
        with pytest.raises(ValidationError):
            ExplanationConfigDTO(
                method=ExplanationMethod.SHAP,
                explanation_type=ExplanationType.LOCAL,
                max_features=101,
            )

    def test_background_samples_validation(self):
        """Test background samples validation."""
        # Valid range
        dto = ExplanationConfigDTO(
            method=ExplanationMethod.SHAP,
            explanation_type=ExplanationType.LOCAL,
            background_samples=1000,
        )
        assert dto.background_samples == 1000

        # Invalid: too small
        with pytest.raises(ValidationError):
            ExplanationConfigDTO(
                method=ExplanationMethod.SHAP,
                explanation_type=ExplanationType.LOCAL,
                background_samples=5,
            )

        # Invalid: too large
        with pytest.raises(ValidationError):
            ExplanationConfigDTO(
                method=ExplanationMethod.SHAP,
                explanation_type=ExplanationType.LOCAL,
                background_samples=10000,
            )

    def test_lime_num_samples_validation(self):
        """Test LIME num samples validation."""
        # Valid range
        dto = ExplanationConfigDTO(
            method=ExplanationMethod.LIME,
            explanation_type=ExplanationType.LOCAL,
            lime_num_samples=2000,
        )
        assert dto.lime_num_samples == 2000

        # Invalid: too small
        with pytest.raises(ValidationError):
            ExplanationConfigDTO(
                method=ExplanationMethod.LIME,
                explanation_type=ExplanationType.LOCAL,
                lime_num_samples=50,
            )

        # Invalid: too large
        with pytest.raises(ValidationError):
            ExplanationConfigDTO(
                method=ExplanationMethod.LIME,
                explanation_type=ExplanationType.LOCAL,
                lime_num_samples=15000,
            )

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            ExplanationConfigDTO(method=ExplanationMethod.SHAP)

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            ExplanationConfigDTO(
                method=ExplanationMethod.SHAP,
                explanation_type=ExplanationType.LOCAL,
                unknown_field="value",
            )


class TestFeatureContributionDTO:
    """Test suite for FeatureContributionDTO."""

    def test_valid_creation(self):
        """Test creating a valid feature contribution DTO."""
        dto = FeatureContributionDTO(
            feature_name="transaction_amount",
            value=1500.0,
            contribution=0.45,
            importance=0.8,
            rank=1,
            description="High transaction amount indicates potential fraud",
            confidence_interval=(0.35, 0.55),
            p_value=0.001,
            normalized_contribution=0.9,
        )

        assert dto.feature_name == "transaction_amount"
        assert dto.value == 1500.0
        assert dto.contribution == 0.45
        assert dto.importance == 0.8
        assert dto.rank == 1
        assert dto.description == "High transaction amount indicates potential fraud"
        assert dto.confidence_interval == (0.35, 0.55)
        assert dto.p_value == 0.001
        assert dto.normalized_contribution == 0.9

    def test_default_values(self):
        """Test default values."""
        dto = FeatureContributionDTO(
            feature_name="feature1",
            value=100.0,
            contribution=0.3,
            importance=0.6,
            rank=2,
        )

        assert dto.description is None
        assert dto.confidence_interval is None
        assert dto.p_value is None
        assert dto.normalized_contribution is None

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            FeatureContributionDTO(
                feature_name="feature1", value=100.0, contribution=0.3, importance=0.6
            )

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            FeatureContributionDTO(
                feature_name="feature1",
                value=100.0,
                contribution=0.3,
                importance=0.6,
                rank=1,
                unknown_field="value",
            )


class TestFeatureInteractionDTO:
    """Test suite for FeatureInteractionDTO."""

    def test_valid_creation(self):
        """Test creating a valid feature interaction DTO."""
        dto = FeatureInteractionDTO(
            feature_1="age",
            feature_2="income",
            interaction_value=0.15,
            interaction_strength=0.7,
            statistical_significance=0.02,
        )

        assert dto.feature_1 == "age"
        assert dto.feature_2 == "income"
        assert dto.interaction_value == 0.15
        assert dto.interaction_strength == 0.7
        assert dto.statistical_significance == 0.02

    def test_default_values(self):
        """Test default values."""
        dto = FeatureInteractionDTO(
            feature_1="feature1",
            feature_2="feature2",
            interaction_value=0.1,
            interaction_strength=0.5,
        )

        assert dto.statistical_significance is None

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            FeatureInteractionDTO(
                feature_1="feature1", feature_2="feature2", interaction_value=0.1
            )

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            FeatureInteractionDTO(
                feature_1="feature1",
                feature_2="feature2",
                interaction_value=0.1,
                interaction_strength=0.5,
                unknown_field="value",
            )


class TestLocalExplanationDTO:
    """Test suite for LocalExplanationDTO."""

    def test_valid_creation(self):
        """Test creating a valid local explanation DTO."""
        instance_id = str(uuid4())
        timestamp = datetime.now()

        feature_contrib = FeatureContributionDTO(
            feature_name="amount",
            value=1000.0,
            contribution=0.4,
            importance=0.8,
            rank=1,
        )

        feature_interaction = FeatureInteractionDTO(
            feature_1="amount",
            feature_2="location",
            interaction_value=0.1,
            interaction_strength=0.6,
        )

        uncertainty = UncertaintyQuantificationDTO(
            epistemic_uncertainty=0.1,
            aleatoric_uncertainty=0.05,
            total_uncertainty=0.15,
            confidence_score=0.85,
            prediction_interval=(0.2, 0.8),
            entropy=0.3,
        )

        config = ExplanationConfigDTO(
            method=ExplanationMethod.SHAP, explanation_type=ExplanationType.LOCAL
        )

        dto = LocalExplanationDTO(
            instance_id=instance_id,
            anomaly_score=0.85,
            prediction="anomaly",
            confidence=0.9,
            feature_contributions=[feature_contrib],
            feature_interactions=[feature_interaction],
            explanation_method=ExplanationMethod.SHAP,
            model_name="IsolationForest",
            timestamp=timestamp,
            baseline_score=0.5,
            counterfactual_examples=[{"amount": 500.0, "location": "safe"}],
            similar_instances=[str(uuid4())],
            explanation_quality_score=0.95,
            uncertainty=uncertainty,
            computation_time=1.5,
            explanation_config=config,
        )

        assert dto.instance_id == instance_id
        assert dto.anomaly_score == 0.85
        assert dto.prediction == "anomaly"
        assert dto.confidence == 0.9
        assert len(dto.feature_contributions) == 1
        assert len(dto.feature_interactions) == 1
        assert dto.explanation_method == ExplanationMethod.SHAP
        assert dto.model_name == "IsolationForest"
        assert dto.timestamp == timestamp
        assert dto.baseline_score == 0.5
        assert len(dto.counterfactual_examples) == 1
        assert len(dto.similar_instances) == 1
        assert dto.explanation_quality_score == 0.95
        assert dto.uncertainty == uncertainty
        assert dto.computation_time == 1.5
        assert dto.explanation_config == config

    def test_default_values(self):
        """Test default values."""
        instance_id = str(uuid4())
        feature_contrib = FeatureContributionDTO(
            feature_name="feature1",
            value=100.0,
            contribution=0.3,
            importance=0.6,
            rank=1,
        )

        dto = LocalExplanationDTO(
            instance_id=instance_id,
            anomaly_score=0.7,
            prediction="normal",
            confidence=0.8,
            feature_contributions=[feature_contrib],
            explanation_method=ExplanationMethod.LIME,
            model_name="OneClassSVM",
            computation_time=2.0,
        )

        assert dto.feature_interactions is None
        assert dto.baseline_score is None
        assert dto.counterfactual_examples is None
        assert dto.similar_instances is None
        assert dto.explanation_quality_score is None
        assert dto.uncertainty is None
        assert dto.explanation_config is None
        assert isinstance(dto.timestamp, datetime)

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            LocalExplanationDTO(
                instance_id=str(uuid4()),
                anomaly_score=0.7,
                prediction="normal",
                confidence=0.8,
            )

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        feature_contrib = FeatureContributionDTO(
            feature_name="feature1",
            value=100.0,
            contribution=0.3,
            importance=0.6,
            rank=1,
        )

        with pytest.raises(ValidationError):
            LocalExplanationDTO(
                instance_id=str(uuid4()),
                anomaly_score=0.7,
                prediction="normal",
                confidence=0.8,
                feature_contributions=[feature_contrib],
                explanation_method=ExplanationMethod.LIME,
                model_name="OneClassSVM",
                computation_time=2.0,
                unknown_field="value",
            )


class TestGlobalExplanationDTO:
    """Test suite for GlobalExplanationDTO."""

    def test_valid_creation(self):
        """Test creating a valid global explanation DTO."""
        timestamp = datetime.now()
        feature_importances = {"feature1": 0.8, "feature2": 0.6, "feature3": 0.4}
        top_features = ["feature1", "feature2", "feature3"]
        performance = {"accuracy": 0.9, "precision": 0.85, "recall": 0.92}

        feature_interaction = FeatureInteractionDTO(
            feature_1="feature1",
            feature_2="feature2",
            interaction_value=0.15,
            interaction_strength=0.7,
        )

        bias_analysis = BiasAnalysisResultDTO(
            detector_id=uuid4(),
            dataset_id=uuid4(),
            analysis_id=uuid4(),
            protected_attributes=["gender"],
            total_samples=1000,
            group_comparisons={"gender": []},
            bias_metrics=[],
            overall_bias_score=0.1,
            is_fair=True,
            fairness_threshold=0.2,
            execution_time=5.0,
            configuration=BiasAnalysisConfigDTO(
                protected_attributes=["gender"],
                privileged_groups={"gender": ["male"]},
                metrics=[BiasMetric.DEMOGRAPHIC_PARITY],
            ),
        )

        config = ExplanationConfigDTO(
            method=ExplanationMethod.SHAP, explanation_type=ExplanationType.GLOBAL
        )

        dto = GlobalExplanationDTO(
            model_name="IsolationForest",
            feature_importances=feature_importances,
            top_features=top_features,
            explanation_method=ExplanationMethod.SHAP,
            model_performance=performance,
            timestamp=timestamp,
            summary="Model relies heavily on feature1 and feature2",
            feature_statistics={"feature1": {"mean": 0.5, "std": 0.2}},
            feature_interactions_global=[feature_interaction],
            partial_dependence_plots={"feature1": {"x": [0, 1], "y": [0, 0.5]}},
            decision_boundary_data={"boundary": "complex"},
            model_complexity_score=0.6,
            interpretability_score=0.8,
            fairness_assessment=bias_analysis,
            samples_analyzed=1000,
            computation_time=10.0,
            explanation_config=config,
        )

        assert dto.model_name == "IsolationForest"
        assert dto.feature_importances == feature_importances
        assert dto.top_features == top_features
        assert dto.explanation_method == ExplanationMethod.SHAP
        assert dto.model_performance == performance
        assert dto.timestamp == timestamp
        assert dto.summary == "Model relies heavily on feature1 and feature2"
        assert dto.feature_statistics is not None
        assert len(dto.feature_interactions_global) == 1
        assert dto.partial_dependence_plots is not None
        assert dto.decision_boundary_data is not None
        assert dto.model_complexity_score == 0.6
        assert dto.interpretability_score == 0.8
        assert dto.fairness_assessment == bias_analysis
        assert dto.samples_analyzed == 1000
        assert dto.computation_time == 10.0
        assert dto.explanation_config == config

    def test_default_values(self):
        """Test default values."""
        dto = GlobalExplanationDTO(
            model_name="TestModel",
            feature_importances={"feature1": 0.8},
            top_features=["feature1"],
            explanation_method=ExplanationMethod.LIME,
            model_performance={"accuracy": 0.9},
            summary="Test summary",
            samples_analyzed=500,
            computation_time=5.0,
        )

        assert dto.feature_statistics is None
        assert dto.feature_interactions_global is None
        assert dto.partial_dependence_plots is None
        assert dto.decision_boundary_data is None
        assert dto.model_complexity_score is None
        assert dto.interpretability_score is None
        assert dto.fairness_assessment is None
        assert dto.explanation_config is None
        assert isinstance(dto.timestamp, datetime)

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            GlobalExplanationDTO(
                model_name="TestModel",
                feature_importances={"feature1": 0.8},
                top_features=["feature1"],
                explanation_method=ExplanationMethod.LIME,
                model_performance={"accuracy": 0.9},
            )

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            GlobalExplanationDTO(
                model_name="TestModel",
                feature_importances={"feature1": 0.8},
                top_features=["feature1"],
                explanation_method=ExplanationMethod.LIME,
                model_performance={"accuracy": 0.9},
                summary="Test summary",
                samples_analyzed=500,
                computation_time=5.0,
                unknown_field="value",
            )


class TestBiasAnalysisConfigDTO:
    """Test suite for BiasAnalysisConfigDTO."""

    def test_valid_creation(self):
        """Test creating a valid bias analysis config DTO."""
        dto = BiasAnalysisConfigDTO(
            protected_attributes=["gender", "race", "age"],
            privileged_groups={"gender": ["male"], "race": ["white"], "age": ["25-50"]},
            metrics=[BiasMetric.DEMOGRAPHIC_PARITY, BiasMetric.EQUALIZED_ODDS],
            threshold=0.7,
            confidence_level=0.99,
            bootstrap_samples=2000,
            min_group_size=50,
        )

        assert dto.protected_attributes == ["gender", "race", "age"]
        assert dto.privileged_groups["gender"] == ["male"]
        assert dto.privileged_groups["race"] == ["white"]
        assert dto.privileged_groups["age"] == ["25-50"]
        assert BiasMetric.DEMOGRAPHIC_PARITY in dto.metrics
        assert BiasMetric.EQUALIZED_ODDS in dto.metrics
        assert dto.threshold == 0.7
        assert dto.confidence_level == 0.99
        assert dto.bootstrap_samples == 2000
        assert dto.min_group_size == 50

    def test_default_values(self):
        """Test default values."""
        dto = BiasAnalysisConfigDTO(
            protected_attributes=["gender"], privileged_groups={"gender": ["male"]}
        )

        assert dto.metrics == [BiasMetric.DEMOGRAPHIC_PARITY]
        assert dto.threshold == 0.5
        assert dto.confidence_level == 0.95
        assert dto.bootstrap_samples == 1000
        assert dto.min_group_size == 30

    def test_confidence_level_validation(self):
        """Test confidence level validation."""
        # Valid range
        dto = BiasAnalysisConfigDTO(
            protected_attributes=["gender"],
            privileged_groups={"gender": ["male"]},
            confidence_level=0.8,
        )
        assert dto.confidence_level == 0.8

        # Invalid: too low
        with pytest.raises(ValidationError):
            BiasAnalysisConfigDTO(
                protected_attributes=["gender"],
                privileged_groups={"gender": ["male"]},
                confidence_level=0.4,
            )

        # Invalid: too high
        with pytest.raises(ValidationError):
            BiasAnalysisConfigDTO(
                protected_attributes=["gender"],
                privileged_groups={"gender": ["male"]},
                confidence_level=1.0,
            )

    def test_threshold_validation(self):
        """Test threshold validation."""
        # Valid range
        dto = BiasAnalysisConfigDTO(
            protected_attributes=["gender"],
            privileged_groups={"gender": ["male"]},
            threshold=0.8,
        )
        assert dto.threshold == 0.8

        # Invalid: negative
        with pytest.raises(ValidationError):
            BiasAnalysisConfigDTO(
                protected_attributes=["gender"],
                privileged_groups={"gender": ["male"]},
                threshold=-0.1,
            )

        # Invalid: greater than 1
        with pytest.raises(ValidationError):
            BiasAnalysisConfigDTO(
                protected_attributes=["gender"],
                privileged_groups={"gender": ["male"]},
                threshold=1.1,
            )

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            BiasAnalysisConfigDTO(protected_attributes=["gender"])

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            BiasAnalysisConfigDTO(
                protected_attributes=["gender"],
                privileged_groups={"gender": ["male"]},
                unknown_field="value",
            )


class TestTrustAssessmentConfigDTO:
    """Test suite for TrustAssessmentConfigDTO."""

    def test_valid_creation(self):
        """Test creating a valid trust assessment config DTO."""
        dto = TrustAssessmentConfigDTO(
            metrics=[
                TrustMetric.CONSISTENCY,
                TrustMetric.STABILITY,
                TrustMetric.FIDELITY,
            ],
            perturbation_ratio=0.2,
            n_perturbations=200,
            stability_samples=100,
            fidelity_samples=300,
            uncertainty_method="bayesian",
            mc_dropout_samples=200,
        )

        assert TrustMetric.CONSISTENCY in dto.metrics
        assert TrustMetric.STABILITY in dto.metrics
        assert TrustMetric.FIDELITY in dto.metrics
        assert dto.perturbation_ratio == 0.2
        assert dto.n_perturbations == 200
        assert dto.stability_samples == 100
        assert dto.fidelity_samples == 300
        assert dto.uncertainty_method == "bayesian"
        assert dto.mc_dropout_samples == 200

    def test_default_values(self):
        """Test default values."""
        dto = TrustAssessmentConfigDTO()

        assert dto.metrics == [TrustMetric.CONSISTENCY, TrustMetric.STABILITY]
        assert dto.perturbation_ratio == 0.1
        assert dto.n_perturbations == 100
        assert dto.stability_samples == 50
        assert dto.fidelity_samples == 100
        assert dto.uncertainty_method == "monte_carlo"
        assert dto.mc_dropout_samples == 100

    def test_perturbation_ratio_validation(self):
        """Test perturbation ratio validation."""
        # Valid range
        dto = TrustAssessmentConfigDTO(perturbation_ratio=0.3)
        assert dto.perturbation_ratio == 0.3

        # Invalid: too small
        with pytest.raises(ValidationError):
            TrustAssessmentConfigDTO(perturbation_ratio=0.005)

        # Invalid: too large
        with pytest.raises(ValidationError):
            TrustAssessmentConfigDTO(perturbation_ratio=0.6)

    def test_n_perturbations_validation(self):
        """Test n_perturbations validation."""
        # Valid range
        dto = TrustAssessmentConfigDTO(n_perturbations=500)
        assert dto.n_perturbations == 500

        # Invalid: too small
        with pytest.raises(ValidationError):
            TrustAssessmentConfigDTO(n_perturbations=5)

        # Invalid: too large
        with pytest.raises(ValidationError):
            TrustAssessmentConfigDTO(n_perturbations=1500)

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            TrustAssessmentConfigDTO(unknown_field="value")


class TestUncertaintyQuantificationDTO:
    """Test suite for UncertaintyQuantificationDTO."""

    def test_valid_creation(self):
        """Test creating a valid uncertainty quantification DTO."""
        dto = UncertaintyQuantificationDTO(
            epistemic_uncertainty=0.15,
            aleatoric_uncertainty=0.08,
            total_uncertainty=0.23,
            confidence_score=0.77,
            prediction_interval=(0.2, 0.8),
            entropy=0.45,
            mc_dropout_variance=0.12,
            ensemble_variance=0.18,
            bayesian_uncertainty=0.20,
        )

        assert dto.epistemic_uncertainty == 0.15
        assert dto.aleatoric_uncertainty == 0.08
        assert dto.total_uncertainty == 0.23
        assert dto.confidence_score == 0.77
        assert dto.prediction_interval == (0.2, 0.8)
        assert dto.entropy == 0.45
        assert dto.mc_dropout_variance == 0.12
        assert dto.ensemble_variance == 0.18
        assert dto.bayesian_uncertainty == 0.20

    def test_default_values(self):
        """Test default values."""
        dto = UncertaintyQuantificationDTO(
            epistemic_uncertainty=0.1,
            aleatoric_uncertainty=0.05,
            total_uncertainty=0.15,
            confidence_score=0.85,
            prediction_interval=(0.3, 0.7),
            entropy=0.2,
        )

        assert dto.mc_dropout_variance is None
        assert dto.ensemble_variance is None
        assert dto.bayesian_uncertainty is None

    def test_confidence_score_validation(self):
        """Test confidence score validation."""
        # Valid range
        dto = UncertaintyQuantificationDTO(
            epistemic_uncertainty=0.1,
            aleatoric_uncertainty=0.05,
            total_uncertainty=0.15,
            confidence_score=0.9,
            prediction_interval=(0.3, 0.7),
            entropy=0.2,
        )
        assert dto.confidence_score == 0.9

        # Invalid: negative
        with pytest.raises(ValidationError):
            UncertaintyQuantificationDTO(
                epistemic_uncertainty=0.1,
                aleatoric_uncertainty=0.05,
                total_uncertainty=0.15,
                confidence_score=-0.1,
                prediction_interval=(0.3, 0.7),
                entropy=0.2,
            )

        # Invalid: greater than 1
        with pytest.raises(ValidationError):
            UncertaintyQuantificationDTO(
                epistemic_uncertainty=0.1,
                aleatoric_uncertainty=0.05,
                total_uncertainty=0.15,
                confidence_score=1.1,
                prediction_interval=(0.3, 0.7),
                entropy=0.2,
            )

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            UncertaintyQuantificationDTO(
                epistemic_uncertainty=0.1,
                aleatoric_uncertainty=0.05,
                total_uncertainty=0.15,
            )

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            UncertaintyQuantificationDTO(
                epistemic_uncertainty=0.1,
                aleatoric_uncertainty=0.05,
                total_uncertainty=0.15,
                confidence_score=0.85,
                prediction_interval=(0.3, 0.7),
                entropy=0.2,
                unknown_field="value",
            )


class TestVisualizationDataDTO:
    """Test suite for VisualizationDataDTO."""

    def test_valid_creation(self):
        """Test creating a valid visualization data DTO."""
        created_at = datetime.now()
        data = {"x": [1, 2, 3], "y": [0.1, 0.5, 0.9]}
        layout = {"title": "Feature Importance", "xaxis": {"title": "Features"}}
        style = {"theme": "dark", "colors": ["blue", "red", "green"]}

        dto = VisualizationDataDTO(
            visualization_type=VisualizationType.FEATURE_IMPORTANCE,
            title="Feature Importance Plot",
            data=data,
            layout=layout,
            style=style,
            created_at=created_at,
            dimensions=(800, 600),
            interactive=True,
            export_formats=["png", "svg", "html", "pdf"],
        )

        assert dto.visualization_type == VisualizationType.FEATURE_IMPORTANCE
        assert dto.title == "Feature Importance Plot"
        assert dto.data == data
        assert dto.layout == layout
        assert dto.style == style
        assert dto.created_at == created_at
        assert dto.dimensions == (800, 600)
        assert dto.interactive is True
        assert dto.export_formats == ["png", "svg", "html", "pdf"]

    def test_default_values(self):
        """Test default values."""
        dto = VisualizationDataDTO(
            visualization_type=VisualizationType.WATERFALL,
            title="Waterfall Chart",
            data={"values": [0.1, 0.2, 0.3]},
        )

        assert dto.layout == {}
        assert dto.style == {}
        assert dto.dimensions is None
        assert dto.interactive is True
        assert dto.export_formats == ["png", "svg", "html"]
        assert isinstance(dto.created_at, datetime)

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            VisualizationDataDTO(
                visualization_type=VisualizationType.HEATMAP, title="Test Chart"
            )

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            VisualizationDataDTO(
                visualization_type=VisualizationType.BAR_CHART,
                title="Bar Chart",
                data={"values": [1, 2, 3]},
                unknown_field="value",
            )


class TestComprehensiveExplanationRequestDTO:
    """Test suite for ComprehensiveExplanationRequestDTO."""

    def test_valid_creation(self):
        """Test creating a valid comprehensive explanation request DTO."""
        detector_id = uuid4()
        dataset_id = uuid4()
        request_id = uuid4()

        explanation_config = ExplanationConfigDTO(
            method=ExplanationMethod.SHAP, explanation_type=ExplanationType.LOCAL
        )

        bias_config = BiasAnalysisConfigDTO(
            protected_attributes=["gender"], privileged_groups={"gender": ["male"]}
        )

        trust_config = TrustAssessmentConfigDTO()

        report_config = ReportGenerationConfigDTO(report_type="detailed", format="html")

        dto = ComprehensiveExplanationRequestDTO(
            detector_id=detector_id,
            dataset_id=dataset_id,
            instance_data={"feature1": 100, "feature2": 200},
            instance_indices=[0, 1, 2],
            explanation_config=explanation_config,
            include_bias_analysis=True,
            bias_config=bias_config,
            include_trust_assessment=True,
            trust_config=trust_config,
            generate_visualizations=True,
            visualization_types=[
                VisualizationType.FEATURE_IMPORTANCE,
                VisualizationType.WATERFALL,
            ],
            generate_report=True,
            report_config=report_config,
            compare_methods=True,
            methods_to_compare=[ExplanationMethod.SHAP, ExplanationMethod.LIME],
            include_cohort_analysis=True,
            request_id=request_id,
            user_id="user123",
            session_id="session456",
            priority="high",
        )

        assert dto.detector_id == detector_id
        assert dto.dataset_id == dataset_id
        assert dto.instance_data == {"feature1": 100, "feature2": 200}
        assert dto.instance_indices == [0, 1, 2]
        assert dto.explanation_config == explanation_config
        assert dto.include_bias_analysis is True
        assert dto.bias_config == bias_config
        assert dto.include_trust_assessment is True
        assert dto.trust_config == trust_config
        assert dto.generate_visualizations is True
        assert dto.visualization_types == [
            VisualizationType.FEATURE_IMPORTANCE,
            VisualizationType.WATERFALL,
        ]
        assert dto.generate_report is True
        assert dto.report_config == report_config
        assert dto.compare_methods is True
        assert dto.methods_to_compare == [
            ExplanationMethod.SHAP,
            ExplanationMethod.LIME,
        ]
        assert dto.include_cohort_analysis is True
        assert dto.request_id == request_id
        assert dto.user_id == "user123"
        assert dto.session_id == "session456"
        assert dto.priority == "high"

    def test_default_values(self):
        """Test default values."""
        detector_id = uuid4()
        explanation_config = ExplanationConfigDTO(
            method=ExplanationMethod.SHAP, explanation_type=ExplanationType.LOCAL
        )

        dto = ComprehensiveExplanationRequestDTO(
            detector_id=detector_id, explanation_config=explanation_config
        )

        assert dto.dataset_id is None
        assert dto.instance_data is None
        assert dto.instance_indices is None
        assert dto.include_bias_analysis is False
        assert dto.bias_config is None
        assert dto.include_trust_assessment is False
        assert dto.trust_config is None
        assert dto.generate_visualizations is True
        assert dto.visualization_types is None
        assert dto.generate_report is False
        assert dto.report_config is None
        assert dto.compare_methods is False
        assert dto.methods_to_compare is None
        assert dto.include_cohort_analysis is False
        assert dto.request_id is None
        assert dto.user_id is None
        assert dto.session_id is None
        assert dto.priority == "medium"

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            ComprehensiveExplanationRequestDTO(detector_id=uuid4())

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        detector_id = uuid4()
        explanation_config = ExplanationConfigDTO(
            method=ExplanationMethod.SHAP, explanation_type=ExplanationType.LOCAL
        )

        with pytest.raises(ValidationError):
            ComprehensiveExplanationRequestDTO(
                detector_id=detector_id,
                explanation_config=explanation_config,
                unknown_field="value",
            )


class TestComprehensiveExplanationResponseDTO:
    """Test suite for ComprehensiveExplanationResponseDTO."""

    def test_valid_creation(self):
        """Test creating a valid comprehensive explanation response DTO."""
        request_id = uuid4()
        timestamp = datetime.now()

        feature_contrib = FeatureContributionDTO(
            feature_name="feature1",
            value=100.0,
            contribution=0.3,
            importance=0.6,
            rank=1,
        )

        local_explanation = LocalExplanationDTO(
            instance_id=str(uuid4()),
            anomaly_score=0.7,
            prediction="anomaly",
            confidence=0.8,
            feature_contributions=[feature_contrib],
            explanation_method=ExplanationMethod.SHAP,
            model_name="IsolationForest",
            computation_time=1.0,
        )

        global_explanation = GlobalExplanationDTO(
            model_name="IsolationForest",
            feature_importances={"feature1": 0.8},
            top_features=["feature1"],
            explanation_method=ExplanationMethod.SHAP,
            model_performance={"accuracy": 0.9},
            summary="Global explanation summary",
            samples_analyzed=1000,
            computation_time=5.0,
        )

        dto = ComprehensiveExplanationResponseDTO(
            request_id=request_id,
            success=True,
            timestamp=timestamp,
            local_explanations=[local_explanation],
            global_explanation=global_explanation,
            total_execution_time=10.0,
            individual_timings={"local": 1.0, "global": 5.0, "visualizations": 4.0},
            memory_usage_mb=512.0,
            warnings=["Low confidence on some predictions"],
            errors=None,
            partial_success=False,
            explanation_quality_scores={"local": 0.9, "global": 0.85},
            confidence_scores={"local": 0.8, "global": 0.9},
        )

        assert dto.request_id == request_id
        assert dto.success is True
        assert dto.timestamp == timestamp
        assert len(dto.local_explanations) == 1
        assert dto.global_explanation == global_explanation
        assert dto.total_execution_time == 10.0
        assert dto.individual_timings == {
            "local": 1.0,
            "global": 5.0,
            "visualizations": 4.0,
        }
        assert dto.memory_usage_mb == 512.0
        assert dto.warnings == ["Low confidence on some predictions"]
        assert dto.errors is None
        assert dto.partial_success is False
        assert dto.explanation_quality_scores == {"local": 0.9, "global": 0.85}
        assert dto.confidence_scores == {"local": 0.8, "global": 0.9}

    def test_default_values(self):
        """Test default values."""
        request_id = uuid4()

        dto = ComprehensiveExplanationResponseDTO(
            request_id=request_id, success=True, total_execution_time=5.0
        )

        assert dto.local_explanations is None
        assert dto.global_explanation is None
        assert dto.cohort_explanations is None
        assert dto.bias_analysis is None
        assert dto.trust_assessment is None
        assert dto.method_comparison is None
        assert dto.visualizations is None
        assert dto.report is None
        assert dto.individual_timings == {}
        assert dto.memory_usage_mb is None
        assert dto.warnings == []
        assert dto.errors is None
        assert dto.partial_success is False
        assert dto.explanation_quality_scores is None
        assert dto.confidence_scores is None
        assert isinstance(dto.timestamp, datetime)

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            ComprehensiveExplanationResponseDTO(request_id=uuid4(), success=True)

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            ComprehensiveExplanationResponseDTO(
                request_id=uuid4(),
                success=True,
                total_execution_time=5.0,
                unknown_field="value",
            )


class TestUtilityFunctions:
    """Test suite for utility functions."""

    def test_create_explanation_config_from_legacy(self):
        """Test creating explanation config from legacy parameters."""
        config = create_explanation_config_from_legacy(
            explanation_method="lime",
            max_features=15,
            background_samples=200,
            random_state=123,
        )

        assert config.method == ExplanationMethod.LIME
        assert config.explanation_type == ExplanationType.LOCAL
        assert config.max_features == 15
        assert config.background_samples == 200
        assert config.random_state == 123

    def test_merge_feature_contributions_mean(self):
        """Test merging feature contributions using mean method."""
        contrib1 = FeatureContributionDTO(
            feature_name="feature1",
            value=100.0,
            contribution=0.3,
            importance=0.6,
            rank=1,
        )

        contrib2 = FeatureContributionDTO(
            feature_name="feature1",
            value=120.0,
            contribution=0.5,
            importance=0.8,
            rank=1,
        )

        contrib3 = FeatureContributionDTO(
            feature_name="feature2",
            value=200.0,
            contribution=0.2,
            importance=0.4,
            rank=2,
        )

        merged = merge_feature_contributions(
            [contrib1, contrib2, contrib3], method="mean"
        )

        assert len(merged) == 2

        # Find feature1 contribution
        feature1_contrib = next(c for c in merged if c.feature_name == "feature1")
        assert feature1_contrib.value == 110.0  # (100 + 120) / 2
        assert feature1_contrib.contribution == 0.4  # (0.3 + 0.5) / 2
        assert feature1_contrib.importance == 0.7  # (0.6 + 0.8) / 2

        # Find feature2 contribution
        feature2_contrib = next(c for c in merged if c.feature_name == "feature2")
        assert feature2_contrib.value == 200.0
        assert feature2_contrib.contribution == 0.2
        assert feature2_contrib.importance == 0.4

    def test_merge_feature_contributions_median(self):
        """Test merging feature contributions using median method."""
        contrib1 = FeatureContributionDTO(
            feature_name="feature1",
            value=100.0,
            contribution=0.2,
            importance=0.4,
            rank=1,
        )

        contrib2 = FeatureContributionDTO(
            feature_name="feature1",
            value=120.0,
            contribution=0.3,
            importance=0.6,
            rank=1,
        )

        contrib3 = FeatureContributionDTO(
            feature_name="feature1",
            value=110.0,
            contribution=0.5,
            importance=0.8,
            rank=1,
        )

        merged = merge_feature_contributions(
            [contrib1, contrib2, contrib3], method="median"
        )

        assert len(merged) == 1
        feature1_contrib = merged[0]
        assert feature1_contrib.value == 110.0  # Median of [100, 110, 120]
        assert feature1_contrib.contribution == 0.3  # Median of [0.2, 0.3, 0.5]
        assert feature1_contrib.importance == 0.6  # Median of [0.4, 0.6, 0.8]

    def test_calculate_explanation_quality_score(self):
        """Test calculating explanation quality score."""
        feature_contrib = FeatureContributionDTO(
            feature_name="feature1",
            value=100.0,
            contribution=0.3,
            importance=0.6,
            rank=1,
        )

        explanation = LocalExplanationDTO(
            instance_id=str(uuid4()),
            anomaly_score=0.7,
            prediction="anomaly",
            confidence=0.8,
            feature_contributions=[feature_contrib],
            explanation_method=ExplanationMethod.SHAP,
            model_name="IsolationForest",
            computation_time=1.0,
        )

        trust_assessment = TrustAssessmentResultDTO(
            detector_id=uuid4(),
            assessment_id=uuid4(),
            trust_metrics=[],
            overall_trust_score=0.85,
            trust_level="high",
            execution_time=2.0,
            configuration=TrustAssessmentConfigDTO(),
        )

        quality_score = calculate_explanation_quality_score(
            explanation, trust_assessment=trust_assessment
        )

        # Base score: 0.7
        # Confidence bonus: (0.8 - 0.5) * 0.2 = 0.06
        # Trust bonus: (0.85 - 0.5) * 0.3 = 0.105
        # Feature coverage: min(1/10, 1.0) * 0.1 = 0.01
        # Total: 0.7 + 0.06 + 0.105 + 0.01 = 0.875
        assert abs(quality_score - 0.875) < 0.001

    def test_create_visualization_config(self):
        """Test creating visualization configuration."""
        config = create_visualization_config(
            visualization_types=[
                VisualizationType.FEATURE_IMPORTANCE,
                VisualizationType.HEATMAP,
            ],
            interactive=False,
            export_formats=["png", "pdf"],
        )

        assert config["types"] == ["feature_importance", "heatmap"]
        assert config["interactive"] is False
        assert config["export_formats"] == ["png", "pdf"]
        assert "style" in config
        assert config["style"]["theme"] == "default"

    def test_validate_explanation_request(self):
        """Test validating explanation request."""
        # Valid request
        valid_request = ComprehensiveExplanationRequestDTO(
            detector_id=uuid4(),
            dataset_id=uuid4(),
            explanation_config=ExplanationConfigDTO(
                method=ExplanationMethod.SHAP, explanation_type=ExplanationType.LOCAL
            ),
        )

        errors = validate_explanation_request(valid_request)
        assert len(errors) == 0

        # Invalid request - missing detector_id
        invalid_request = ComprehensiveExplanationRequestDTO(
            detector_id=None,
            explanation_config=ExplanationConfigDTO(
                method=ExplanationMethod.SHAP, explanation_type=ExplanationType.LOCAL
            ),
        )

        # This would fail at the Pydantic validation level, so we test other validation

        # Invalid request - bias analysis without config
        bias_request = ComprehensiveExplanationRequestDTO(
            detector_id=uuid4(),
            dataset_id=uuid4(),
            explanation_config=ExplanationConfigDTO(
                method=ExplanationMethod.SHAP, explanation_type=ExplanationType.LOCAL
            ),
            include_bias_analysis=True,
        )

        errors = validate_explanation_request(bias_request)
        assert "bias_config is required when include_bias_analysis is True" in errors


class TestExplanationDTOIntegration:
    """Test integration scenarios for explanation DTOs."""

    def test_complete_explanation_workflow(self):
        """Test complete explanation workflow."""
        # Create comprehensive explanation request
        detector_id = uuid4()
        dataset_id = uuid4()

        explanation_config = ExplanationConfigDTO(
            method=ExplanationMethod.SHAP,
            explanation_type=ExplanationType.LOCAL,
            max_features=10,
            background_samples=100,
        )

        bias_config = BiasAnalysisConfigDTO(
            protected_attributes=["gender", "age"],
            privileged_groups={"gender": ["male"], "age": ["25-50"]},
            metrics=[BiasMetric.DEMOGRAPHIC_PARITY],
        )

        trust_config = TrustAssessmentConfigDTO(
            metrics=[TrustMetric.CONSISTENCY, TrustMetric.STABILITY]
        )

        request = ComprehensiveExplanationRequestDTO(
            detector_id=detector_id,
            dataset_id=dataset_id,
            explanation_config=explanation_config,
            include_bias_analysis=True,
            bias_config=bias_config,
            include_trust_assessment=True,
            trust_config=trust_config,
            generate_visualizations=True,
        )

        # Create response components
        feature_contrib = FeatureContributionDTO(
            feature_name="transaction_amount",
            value=1500.0,
            contribution=0.45,
            importance=0.8,
            rank=1,
        )

        local_explanation = LocalExplanationDTO(
            instance_id=str(uuid4()),
            anomaly_score=0.85,
            prediction="anomaly",
            confidence=0.9,
            feature_contributions=[feature_contrib],
            explanation_method=ExplanationMethod.SHAP,
            model_name="IsolationForest",
            computation_time=2.0,
        )

        trust_metric = TrustMetricResultDTO(
            metric_name=TrustMetric.CONSISTENCY,
            score=0.85,
            interpretation="High consistency",
            confidence_level="high",
        )

        trust_assessment = TrustAssessmentResultDTO(
            detector_id=detector_id,
            assessment_id=uuid4(),
            trust_metrics=[trust_metric],
            overall_trust_score=0.85,
            trust_level="high",
            execution_time=3.0,
            configuration=trust_config,
        )

        # Create comprehensive response
        response = ComprehensiveExplanationResponseDTO(
            request_id=request.request_id or uuid4(),
            success=True,
            local_explanations=[local_explanation],
            trust_assessment=trust_assessment,
            total_execution_time=5.0,
            individual_timings={"local": 2.0, "trust": 3.0},
        )

        # Verify workflow consistency
        assert response.success is True
        assert len(response.local_explanations) == 1
        assert response.trust_assessment == trust_assessment
        assert response.total_execution_time == 5.0
        assert response.individual_timings["local"] == 2.0
        assert response.individual_timings["trust"] == 3.0

    def test_dto_serialization(self):
        """Test DTO serialization and deserialization."""
        # Create feature contribution DTO
        original_contrib = FeatureContributionDTO(
            feature_name="amount",
            value=1000.0,
            contribution=0.4,
            importance=0.8,
            rank=1,
            description="Transaction amount feature",
        )

        # Serialize to dict
        contrib_dict = original_contrib.model_dump()

        assert contrib_dict["feature_name"] == "amount"
        assert contrib_dict["value"] == 1000.0
        assert contrib_dict["contribution"] == 0.4
        assert contrib_dict["importance"] == 0.8
        assert contrib_dict["rank"] == 1
        assert contrib_dict["description"] == "Transaction amount feature"

        # Deserialize from dict
        restored_contrib = FeatureContributionDTO.model_validate(contrib_dict)

        assert restored_contrib.feature_name == original_contrib.feature_name
        assert restored_contrib.value == original_contrib.value
        assert restored_contrib.contribution == original_contrib.contribution
        assert restored_contrib.importance == original_contrib.importance
        assert restored_contrib.rank == original_contrib.rank
        assert restored_contrib.description == original_contrib.description

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Minimum max_features
        config = ExplanationConfigDTO(
            method=ExplanationMethod.SHAP,
            explanation_type=ExplanationType.LOCAL,
            max_features=1,
        )
        assert config.max_features == 1

        # Maximum max_features
        config = ExplanationConfigDTO(
            method=ExplanationMethod.SHAP,
            explanation_type=ExplanationType.LOCAL,
            max_features=100,
        )
        assert config.max_features == 100

        # Zero contribution
        contrib = FeatureContributionDTO(
            feature_name="zero_contrib",
            value=0.0,
            contribution=0.0,
            importance=0.0,
            rank=10,
        )
        assert contrib.contribution == 0.0
        assert contrib.importance == 0.0

        # Negative contribution
        contrib = FeatureContributionDTO(
            feature_name="negative_contrib",
            value=-100.0,
            contribution=-0.5,
            importance=0.5,
            rank=1,
        )
        assert contrib.contribution == -0.5
        assert contrib.value == -100.0

    def test_enum_validations(self):
        """Test enum validations."""
        # Valid explanation method
        config = ExplanationConfigDTO(
            method=ExplanationMethod.LIME, explanation_type=ExplanationType.GLOBAL
        )
        assert config.method == ExplanationMethod.LIME

        # Valid explanation type
        assert config.explanation_type == ExplanationType.GLOBAL

        # Valid bias metric
        bias_config = BiasAnalysisConfigDTO(
            protected_attributes=["gender"],
            privileged_groups={"gender": ["male"]},
            metrics=[BiasMetric.EQUALIZED_ODDS],
        )
        assert BiasMetric.EQUALIZED_ODDS in bias_config.metrics

        # Valid trust metric
        trust_config = TrustAssessmentConfigDTO(metrics=[TrustMetric.ROBUSTNESS])
        assert TrustMetric.ROBUSTNESS in trust_config.metrics

        # Valid visualization type
        viz = VisualizationDataDTO(
            visualization_type=VisualizationType.CORRELATION_MATRIX,
            title="Correlation Matrix",
            data={"matrix": [[1, 0.5], [0.5, 1]]},
        )
        assert viz.visualization_type == VisualizationType.CORRELATION_MATRIX
