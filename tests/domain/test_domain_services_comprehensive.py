"""Comprehensive tests for domain services.

This module provides comprehensive test coverage for all domain services
including threshold calculation, feature validation, ensemble aggregation,
anomaly scoring, and explainability services.
"""

import numpy as np
import pandas as pd
import pytest

from pynomaly.domain.entities import Anomaly, Dataset
from pynomaly.domain.exceptions import InvalidValueError, ValidationError
from pynomaly.domain.services.anomaly_scorer import AnomalyScorer
from pynomaly.domain.services.ensemble_aggregator import EnsembleAggregator
from pynomaly.domain.services.explainability_service import ExplainabilityService
from pynomaly.domain.services.feature_validator import FeatureValidator
from pynomaly.domain.services.threshold_calculator import ThresholdCalculator
from pynomaly.domain.value_objects import (
    AnomalyScore,
    ConfidenceInterval,
    ContaminationRate,
)


class TestThresholdCalculator:
    """Comprehensive tests for ThresholdCalculator domain service."""

    @pytest.fixture
    def threshold_calculator(self):
        """Create threshold calculator instance."""
        return ThresholdCalculator()

    @pytest.fixture
    def sample_scores(self):
        """Create sample anomaly scores for testing."""
        return [
            AnomalyScore(0.1),
            AnomalyScore(0.2),
            AnomalyScore(0.3),
            AnomalyScore(0.4),
            AnomalyScore(0.5),
            AnomalyScore(0.6),
            AnomalyScore(0.7),
            AnomalyScore(0.8),
            AnomalyScore(0.9),
            AnomalyScore(0.95),
        ]

    def test_calculate_percentile_threshold(self, threshold_calculator, sample_scores):
        """Test percentile-based threshold calculation."""
        threshold = threshold_calculator.calculate_percentile_threshold(
            scores=sample_scores, percentile=90.0
        )

        assert isinstance(threshold, float)
        assert 0.0 <= threshold <= 1.0
        # 90th percentile of our scores should be around 0.9
        assert abs(threshold - 0.9) < 0.1

    def test_calculate_contamination_threshold(
        self, threshold_calculator, sample_scores
    ):
        """Test contamination-based threshold calculation."""
        contamination_rate = ContaminationRate(0.1)  # 10% contamination

        threshold = threshold_calculator.calculate_contamination_threshold(
            scores=sample_scores, contamination_rate=contamination_rate
        )

        assert isinstance(threshold, float)
        assert 0.0 <= threshold <= 1.0
        # With 10% contamination and 10 scores, threshold should be around the 9th highest score
        expected_threshold = sorted([s.value for s in sample_scores], reverse=True)[
            0
        ]  # 90th percentile
        assert abs(threshold - expected_threshold) < 0.1

    def test_calculate_statistical_threshold(self, threshold_calculator, sample_scores):
        """Test statistical-based threshold calculation."""
        threshold = threshold_calculator.calculate_statistical_threshold(
            scores=sample_scores, method="iqr", multiplier=1.5
        )

        assert isinstance(threshold, float)
        assert 0.0 <= threshold <= 1.0

        # Test z-score method
        z_threshold = threshold_calculator.calculate_statistical_threshold(
            scores=sample_scores, method="zscore", multiplier=2.0
        )

        assert isinstance(z_threshold, float)
        assert 0.0 <= z_threshold <= 1.0

    def test_calculate_adaptive_threshold(self, threshold_calculator, sample_scores):
        """Test adaptive threshold calculation."""
        # Create historical scores to simulate learning
        historical_scores = [
            [AnomalyScore(0.1 + i * 0.05) for i in range(10)]
            for _ in range(5)  # 5 historical periods
        ]

        threshold = threshold_calculator.calculate_adaptive_threshold(
            current_scores=sample_scores,
            historical_scores=historical_scores,
            adaptation_rate=0.1,
        )

        assert isinstance(threshold, float)
        assert 0.0 <= threshold <= 1.0

    def test_calculate_threshold_with_confidence(
        self, threshold_calculator, sample_scores
    ):
        """Test threshold calculation with confidence intervals."""
        (
            threshold,
            confidence,
        ) = threshold_calculator.calculate_threshold_with_confidence(
            scores=sample_scores,
            method="percentile",
            percentile=95.0,
            confidence_level=0.95,
        )

        assert isinstance(threshold, float)
        assert isinstance(confidence, ConfidenceInterval)
        assert confidence.contains(threshold)
        assert confidence.confidence_level == 0.95

    def test_threshold_validation(self, threshold_calculator):
        """Test threshold calculator validation."""
        with pytest.raises(ValidationError):
            threshold_calculator.calculate_percentile_threshold(
                scores=[],  # Empty scores
                percentile=50.0,
            )

        with pytest.raises(InvalidValueError):
            threshold_calculator.calculate_percentile_threshold(
                scores=[AnomalyScore(0.5)],
                percentile=101.0,  # Invalid percentile
            )

    def test_threshold_optimization(self, threshold_calculator, sample_scores):
        """Test threshold optimization for specific criteria."""
        # Simulate ground truth for optimization
        ground_truth = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # Last 5 are anomalies

        optimal_threshold = threshold_calculator.optimize_threshold(
            scores=sample_scores, ground_truth=ground_truth, metric="f1_score"
        )

        assert isinstance(optimal_threshold, float)
        assert 0.0 <= optimal_threshold <= 1.0

        # Test different optimization metrics
        precision_threshold = threshold_calculator.optimize_threshold(
            scores=sample_scores, ground_truth=ground_truth, metric="precision"
        )

        recall_threshold = threshold_calculator.optimize_threshold(
            scores=sample_scores, ground_truth=ground_truth, metric="recall"
        )

        assert isinstance(precision_threshold, float)
        assert isinstance(recall_threshold, float)


class TestFeatureValidator:
    """Comprehensive tests for FeatureValidator domain service."""

    @pytest.fixture
    def feature_validator(self):
        """Create feature validator instance."""
        return FeatureValidator()

    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset for validation."""
        data = pd.DataFrame(
            {
                "numeric_feature": [1.0, 2.0, 3.0, 4.0, 5.0],
                "categorical_feature": ["A", "B", "C", "A", "B"],
                "missing_feature": [1.0, 2.0, None, 4.0, 5.0],
                "constant_feature": [1.0, 1.0, 1.0, 1.0, 1.0],
                "outlier_feature": [1.0, 2.0, 3.0, 100.0, 5.0],
            }
        )
        return Dataset(name="Test Dataset", data=data)

    def test_validate_feature_types(self, feature_validator, sample_dataset):
        """Test feature type validation."""
        validation_result = feature_validator.validate_feature_types(sample_dataset)

        assert "numeric_features" in validation_result
        assert "categorical_features" in validation_result
        assert "mixed_features" in validation_result
        assert "validation_passed" in validation_result

        assert "numeric_feature" in validation_result["numeric_features"]
        assert "categorical_feature" in validation_result["categorical_features"]

    def test_validate_missing_values(self, feature_validator, sample_dataset):
        """Test missing value validation."""
        validation_result = feature_validator.validate_missing_values(
            dataset=sample_dataset, max_missing_ratio=0.3
        )

        assert "missing_counts" in validation_result
        assert "missing_ratios" in validation_result
        assert "validation_passed" in validation_result
        assert "features_exceeding_threshold" in validation_result

        # missing_feature has 1/5 = 20% missing, which is below 30% threshold
        assert validation_result["validation_passed"]
        assert len(validation_result["features_exceeding_threshold"]) == 0

    def test_validate_feature_distribution(self, feature_validator, sample_dataset):
        """Test feature distribution validation."""
        validation_result = feature_validator.validate_feature_distribution(
            dataset=sample_dataset,
            distribution_tests=["normality", "outliers", "variance"],
        )

        assert "distribution_tests" in validation_result
        assert "outliers_detected" in validation_result
        assert "validation_summary" in validation_result

        # outlier_feature should be flagged
        assert len(validation_result["outliers_detected"]) > 0

    def test_validate_feature_correlation(self, feature_validator, sample_dataset):
        """Test feature correlation validation."""
        # Add a correlated feature
        sample_dataset.data["correlated_feature"] = (
            sample_dataset.data["numeric_feature"] * 2
        )

        validation_result = feature_validator.validate_feature_correlation(
            dataset=sample_dataset, correlation_threshold=0.95
        )

        assert "correlation_matrix" in validation_result
        assert "highly_correlated_pairs" in validation_result
        assert "validation_passed" in validation_result

        # Should detect correlation between numeric_feature and correlated_feature
        assert len(validation_result["highly_correlated_pairs"]) > 0

    def test_validate_feature_importance(self, feature_validator, sample_dataset):
        """Test feature importance validation."""
        # Create a target variable
        target = np.array([0, 1, 0, 1, 0])

        validation_result = feature_validator.validate_feature_importance(
            dataset=sample_dataset, target=target, importance_threshold=0.01
        )

        assert "feature_importance" in validation_result
        assert "low_importance_features" in validation_result
        assert "validation_passed" in validation_result

        # constant_feature should have low importance
        assert "constant_feature" in validation_result["low_importance_features"]

    def test_comprehensive_validation(self, feature_validator, sample_dataset):
        """Test comprehensive feature validation."""
        validation_config = {
            "check_types": True,
            "check_missing": True,
            "check_distribution": True,
            "check_correlation": True,
            "missing_threshold": 0.2,
            "correlation_threshold": 0.9,
        }

        validation_result = feature_validator.comprehensive_validation(
            dataset=sample_dataset, config=validation_config
        )

        assert "overall_score" in validation_result
        assert "validation_details" in validation_result
        assert "recommendations" in validation_result
        assert "issues_found" in validation_result

        assert 0.0 <= validation_result["overall_score"] <= 1.0
        assert isinstance(validation_result["recommendations"], list)

    def test_feature_validation_with_remediation(
        self, feature_validator, sample_dataset
    ):
        """Test feature validation with automatic remediation suggestions."""
        validation_result = feature_validator.validate_with_remediation(
            dataset=sample_dataset, auto_suggest_fixes=True
        )

        assert "validation_issues" in validation_result
        assert "remediation_suggestions" in validation_result
        assert "estimated_improvement" in validation_result

        # Should suggest removing constant features
        constant_suggestions = [
            s
            for s in validation_result["remediation_suggestions"]
            if "constant" in s.get("issue_type", "").lower()
        ]
        assert len(constant_suggestions) > 0


class TestEnsembleAggregator:
    """Comprehensive tests for EnsembleAggregator domain service."""

    @pytest.fixture
    def ensemble_aggregator(self):
        """Create ensemble aggregator instance."""
        return EnsembleAggregator()

    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions from multiple detectors."""
        return {
            "detector_1": [AnomalyScore(0.8), AnomalyScore(0.2), AnomalyScore(0.9)],
            "detector_2": [AnomalyScore(0.7), AnomalyScore(0.3), AnomalyScore(0.8)],
            "detector_3": [AnomalyScore(0.9), AnomalyScore(0.1), AnomalyScore(0.7)],
        }

    def test_voting_aggregation(self, ensemble_aggregator, sample_predictions):
        """Test voting-based ensemble aggregation."""
        threshold = 0.5
        aggregated_result = ensemble_aggregator.aggregate_voting(
            predictions=sample_predictions,
            threshold=threshold,
            voting_strategy="majority",
        )

        assert "ensemble_scores" in aggregated_result
        assert "ensemble_labels" in aggregated_result
        assert "voting_details" in aggregated_result

        ensemble_scores = aggregated_result["ensemble_scores"]
        assert len(ensemble_scores) == 3
        assert all(isinstance(score, AnomalyScore) for score in ensemble_scores)

        # Test different voting strategies
        unanimous_result = ensemble_aggregator.aggregate_voting(
            predictions=sample_predictions,
            threshold=threshold,
            voting_strategy="unanimous",
        )

        assert "ensemble_labels" in unanimous_result

    def test_weighted_averaging(self, ensemble_aggregator, sample_predictions):
        """Test weighted averaging ensemble aggregation."""
        weights = {"detector_1": 0.5, "detector_2": 0.3, "detector_3": 0.2}

        aggregated_result = ensemble_aggregator.aggregate_weighted_average(
            predictions=sample_predictions, weights=weights
        )

        assert "ensemble_scores" in aggregated_result
        assert "weight_distribution" in aggregated_result

        ensemble_scores = aggregated_result["ensemble_scores"]
        assert len(ensemble_scores) == 3

        # Verify weighted calculation for first sample
        expected_first = 0.8 * 0.5 + 0.7 * 0.3 + 0.9 * 0.2
        assert abs(ensemble_scores[0].value - expected_first) < 0.01

    def test_stacking_aggregation(self, ensemble_aggregator, sample_predictions):
        """Test stacking-based ensemble aggregation."""
        # Create training data for meta-learner
        training_predictions = {
            "detector_1": [
                AnomalyScore(0.6),
                AnomalyScore(0.4),
                AnomalyScore(0.8),
                AnomalyScore(0.3),
            ],
            "detector_2": [
                AnomalyScore(0.5),
                AnomalyScore(0.5),
                AnomalyScore(0.7),
                AnomalyScore(0.2),
            ],
            "detector_3": [
                AnomalyScore(0.7),
                AnomalyScore(0.3),
                AnomalyScore(0.9),
                AnomalyScore(0.4),
            ],
        }
        training_labels = [1, 0, 1, 0]

        aggregated_result = ensemble_aggregator.aggregate_stacking(
            predictions=sample_predictions,
            training_predictions=training_predictions,
            training_labels=training_labels,
            meta_learner="logistic_regression",
        )

        assert "ensemble_scores" in aggregated_result
        assert "meta_learner_info" in aggregated_result
        assert "stacking_performance" in aggregated_result

    def test_rank_aggregation(self, ensemble_aggregator, sample_predictions):
        """Test rank-based ensemble aggregation."""
        aggregated_result = ensemble_aggregator.aggregate_ranks(
            predictions=sample_predictions, rank_method="average"
        )

        assert "ensemble_scores" in aggregated_result
        assert "rank_matrix" in aggregated_result
        assert "rank_aggregation_method" in aggregated_result

        # Test different rank methods
        min_rank_result = ensemble_aggregator.aggregate_ranks(
            predictions=sample_predictions, rank_method="min"
        )

        max_rank_result = ensemble_aggregator.aggregate_ranks(
            predictions=sample_predictions, rank_method="max"
        )

        assert "ensemble_scores" in min_rank_result
        assert "ensemble_scores" in max_rank_result

    def test_adaptive_ensemble(self, ensemble_aggregator, sample_predictions):
        """Test adaptive ensemble aggregation."""
        # Performance history for each detector
        performance_history = {
            "detector_1": {"accuracy": 0.85, "precision": 0.8, "recall": 0.9},
            "detector_2": {"accuracy": 0.80, "precision": 0.85, "recall": 0.75},
            "detector_3": {"accuracy": 0.90, "precision": 0.9, "recall": 0.85},
        }

        aggregated_result = ensemble_aggregator.aggregate_adaptive(
            predictions=sample_predictions,
            performance_history=performance_history,
            adaptation_metric="accuracy",
        )

        assert "ensemble_scores" in aggregated_result
        assert "adaptive_weights" in aggregated_result
        assert "performance_based_ranking" in aggregated_result

    def test_uncertainty_aware_ensemble(self, ensemble_aggregator, sample_predictions):
        """Test uncertainty-aware ensemble aggregation."""
        # Add uncertainty information
        predictions_with_uncertainty = {}
        for detector, scores in sample_predictions.items():
            predictions_with_uncertainty[detector] = {
                "scores": scores,
                "uncertainty": [0.1, 0.05, 0.15],  # Uncertainty estimates
            }

        aggregated_result = ensemble_aggregator.aggregate_uncertainty_aware(
            predictions=predictions_with_uncertainty, uncertainty_weighting=True
        )

        assert "ensemble_scores" in aggregated_result
        assert "uncertainty_scores" in aggregated_result
        assert "confidence_intervals" in aggregated_result

    def test_ensemble_diversity_analysis(self, ensemble_aggregator, sample_predictions):
        """Test ensemble diversity analysis."""
        diversity_analysis = ensemble_aggregator.analyze_diversity(
            predictions=sample_predictions,
            diversity_metrics=["disagreement", "correlation", "entropy"],
        )

        assert "diversity_scores" in diversity_analysis
        assert "pairwise_correlations" in diversity_analysis
        assert "overall_diversity" in diversity_analysis
        assert "diversity_recommendations" in diversity_analysis

        assert 0.0 <= diversity_analysis["overall_diversity"] <= 1.0


class TestAnomalyScorer:
    """Comprehensive tests for AnomalyScorer domain service."""

    @pytest.fixture
    def anomaly_scorer(self):
        """Create anomaly scorer instance."""
        return AnomalyScorer()

    @pytest.fixture
    def sample_data(self):
        """Create sample data for scoring."""
        return pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 100],  # Last value is outlier
                "feature2": [2, 4, 6, 8, 200],  # Last value is outlier
                "feature3": [1, 1, 1, 1, 1],  # Constant feature
            }
        )

    def test_isolation_scoring(self, anomaly_scorer, sample_data):
        """Test isolation-based anomaly scoring."""
        scores = anomaly_scorer.calculate_isolation_scores(
            data=sample_data, contamination=0.2
        )

        assert len(scores) == len(sample_data)
        assert all(isinstance(score, AnomalyScore) for score in scores)
        assert all(0.0 <= score.value <= 1.0 for score in scores)

        # Last sample should have high anomaly score
        assert scores[-1].value > scores[0].value

    def test_density_scoring(self, anomaly_scorer, sample_data):
        """Test density-based anomaly scoring."""
        scores = anomaly_scorer.calculate_density_scores(
            data=sample_data, method="local_outlier_factor", n_neighbors=3
        )

        assert len(scores) == len(sample_data)
        assert all(isinstance(score, AnomalyScore) for score in scores)

        # Test different density methods
        knn_scores = anomaly_scorer.calculate_density_scores(
            data=sample_data, method="k_nearest_neighbors", n_neighbors=2
        )

        assert len(knn_scores) == len(sample_data)

    def test_distance_scoring(self, anomaly_scorer, sample_data):
        """Test distance-based anomaly scoring."""
        scores = anomaly_scorer.calculate_distance_scores(
            data=sample_data, distance_metric="euclidean", reference_point="centroid"
        )

        assert len(scores) == len(sample_data)
        assert all(isinstance(score, AnomalyScore) for score in scores)

        # Test different distance metrics
        manhattan_scores = anomaly_scorer.calculate_distance_scores(
            data=sample_data, distance_metric="manhattan", reference_point="median"
        )

        assert len(manhattan_scores) == len(sample_data)

    def test_statistical_scoring(self, anomaly_scorer, sample_data):
        """Test statistical anomaly scoring."""
        scores = anomaly_scorer.calculate_statistical_scores(
            data=sample_data, method="z_score", aggregation="max"
        )

        assert len(scores) == len(sample_data)
        assert all(isinstance(score, AnomalyScore) for score in scores)

        # Test different statistical methods
        iqr_scores = anomaly_scorer.calculate_statistical_scores(
            data=sample_data, method="iqr", aggregation="mean"
        )

        assert len(iqr_scores) == len(sample_data)

    def test_ensemble_scoring(self, anomaly_scorer, sample_data):
        """Test ensemble anomaly scoring."""
        scoring_methods = [
            {"method": "isolation", "weight": 0.4},
            {"method": "density", "weight": 0.3},
            {"method": "statistical", "weight": 0.3},
        ]

        scores = anomaly_scorer.calculate_ensemble_scores(
            data=sample_data,
            methods=scoring_methods,
            aggregation_strategy="weighted_average",
        )

        assert len(scores) == len(sample_data)
        assert all(isinstance(score, AnomalyScore) for score in scores)

        # Test different aggregation strategies
        voting_scores = anomaly_scorer.calculate_ensemble_scores(
            data=sample_data, methods=scoring_methods, aggregation_strategy="voting"
        )

        assert len(voting_scores) == len(sample_data)

    def test_contextual_scoring(self, anomaly_scorer, sample_data):
        """Test contextual anomaly scoring."""
        # Add time component for temporal context
        sample_data["timestamp"] = pd.date_range(
            "2023-01-01", periods=len(sample_data), freq="1H"
        )

        scores = anomaly_scorer.calculate_contextual_scores(
            data=sample_data,
            context_features=["timestamp"],
            behavioral_features=["feature1", "feature2"],
            context_method="temporal_windowing",
        )

        assert len(scores) == len(sample_data)
        assert all(isinstance(score, AnomalyScore) for score in scores)

    def test_score_calibration(self, anomaly_scorer, sample_data):
        """Test anomaly score calibration."""
        # Get raw scores
        raw_scores = anomaly_scorer.calculate_isolation_scores(
            data=sample_data, contamination=0.2
        )

        # Calibrate scores
        calibrated_scores = anomaly_scorer.calibrate_scores(
            scores=raw_scores,
            calibration_method="platt_scaling",
            reference_distribution="normal",
        )

        assert len(calibrated_scores) == len(raw_scores)
        assert all(isinstance(score, AnomalyScore) for score in calibrated_scores)

        # Test different calibration methods
        isotonic_scores = anomaly_scorer.calibrate_scores(
            scores=raw_scores, calibration_method="isotonic_regression"
        )

        assert len(isotonic_scores) == len(raw_scores)


class TestExplainabilityService:
    """Comprehensive tests for ExplainabilityService domain service."""

    @pytest.fixture
    def explainability_service(self):
        """Create explainability service instance."""
        return ExplainabilityService()

    @pytest.fixture
    def sample_anomaly_data(self):
        """Create sample anomaly data for explanation."""
        data = pd.DataFrame(
            {
                "temperature": [20, 21, 22, 85],  # Last value is anomalous
                "pressure": [1000, 1010, 1020, 900],  # Last value is anomalous
                "humidity": [50, 52, 54, 30],  # Last value is anomalous
            }
        )

        anomaly = Anomaly(
            score=AnomalyScore(0.95),
            data_point=data.iloc[-1].to_dict(),
            detector_name="test_detector",
        )

        return data, anomaly

    def test_feature_importance_explanation(
        self, explainability_service, sample_anomaly_data
    ):
        """Test feature importance-based explanations."""
        data, anomaly = sample_anomaly_data

        explanation = explainability_service.explain_feature_importance(
            anomaly=anomaly,
            reference_data=data[:-1],  # All except anomaly
            method="permutation_importance",
        )

        assert "feature_importance" in explanation
        assert "ranking" in explanation
        assert "explanation_text" in explanation

        # Should identify temperature as important contributor
        assert "temperature" in explanation["feature_importance"]
        assert explanation["feature_importance"]["temperature"] > 0

    def test_shap_explanation(self, explainability_service, sample_anomaly_data):
        """Test SHAP-based explanations."""
        data, anomaly = sample_anomaly_data

        explanation = explainability_service.explain_shap(
            anomaly=anomaly, reference_data=data[:-1], model_type="tree_based"
        )

        assert "shap_values" in explanation
        assert "base_value" in explanation
        assert "feature_contributions" in explanation
        assert "explanation_plot_data" in explanation

    def test_lime_explanation(self, explainability_service, sample_anomaly_data):
        """Test LIME-based explanations."""
        data, anomaly = sample_anomaly_data

        explanation = explainability_service.explain_lime(
            anomaly=anomaly, reference_data=data[:-1], num_features=3, num_samples=1000
        )

        assert "lime_explanation" in explanation
        assert "feature_weights" in explanation
        assert "local_model_score" in explanation
        assert "explanation_text" in explanation

    def test_counterfactual_explanation(
        self, explainability_service, sample_anomaly_data
    ):
        """Test counterfactual explanations."""
        data, anomaly = sample_anomaly_data

        explanation = explainability_service.explain_counterfactual(
            anomaly=anomaly,
            reference_data=data[:-1],
            max_iterations=100,
            distance_metric="euclidean",
        )

        assert "counterfactual_instance" in explanation
        assert "feature_changes" in explanation
        assert "distance_to_counterfactual" in explanation
        assert "explanation_text" in explanation

    def test_rule_based_explanation(self, explainability_service, sample_anomaly_data):
        """Test rule-based explanations."""
        data, anomaly = sample_anomaly_data

        explanation = explainability_service.explain_rules(
            anomaly=anomaly,
            reference_data=data[:-1],
            rule_extraction_method="decision_tree",
            max_depth=5,
        )

        assert "rules" in explanation
        assert "rule_coverage" in explanation
        assert "rule_confidence" in explanation
        assert "explanation_text" in explanation

    def test_temporal_explanation(self, explainability_service):
        """Test temporal explanations for time series anomalies."""
        # Create time series data
        dates = pd.date_range("2023-01-01", periods=100, freq="1H")
        values = np.sin(np.linspace(0, 4 * np.pi, 100)) + np.random.normal(0, 0.1, 100)
        values[95] = 5.0  # Inject anomaly

        ts_data = pd.DataFrame({"timestamp": dates, "value": values})

        anomaly = Anomaly(
            score=AnomalyScore(0.9),
            data_point={"timestamp": dates[95], "value": values[95]},
            detector_name="time_series_detector",
        )

        explanation = explainability_service.explain_temporal(
            anomaly=anomaly,
            time_series_data=ts_data,
            window_size=10,
            seasonality_check=True,
        )

        assert "temporal_context" in explanation
        assert "seasonal_deviation" in explanation
        assert "trend_analysis" in explanation
        assert "explanation_text" in explanation

    def test_comparative_explanation(self, explainability_service, sample_anomaly_data):
        """Test comparative explanations."""
        data, anomaly = sample_anomaly_data

        # Create a normal instance for comparison
        normal_instance = data.iloc[0].to_dict()

        explanation = explainability_service.explain_comparative(
            anomaly=anomaly,
            normal_instance=normal_instance,
            comparison_metrics=["difference", "ratio", "percentile"],
        )

        assert "feature_comparisons" in explanation
        assert "similarity_score" in explanation
        assert "key_differences" in explanation
        assert "explanation_text" in explanation

    def test_ensemble_explanation(self, explainability_service, sample_anomaly_data):
        """Test ensemble explanations combining multiple methods."""
        data, anomaly = sample_anomaly_data

        explanation = explainability_service.explain_ensemble(
            anomaly=anomaly,
            reference_data=data[:-1],
            methods=["feature_importance", "lime", "rules"],
            aggregation_strategy="weighted_vote",
        )

        assert "individual_explanations" in explanation
        assert "ensemble_ranking" in explanation
        assert "confidence_score" in explanation
        assert "consolidated_explanation" in explanation

    def test_explanation_quality_assessment(
        self, explainability_service, sample_anomaly_data
    ):
        """Test explanation quality assessment."""
        data, anomaly = sample_anomaly_data

        # Get an explanation first
        explanation = explainability_service.explain_feature_importance(
            anomaly=anomaly, reference_data=data[:-1]
        )

        quality_assessment = explainability_service.assess_explanation_quality(
            explanation=explanation,
            quality_metrics=["consistency", "completeness", "contrastivity"],
        )

        assert "quality_scores" in quality_assessment
        assert "overall_quality" in quality_assessment
        assert "improvement_suggestions" in quality_assessment

        assert 0.0 <= quality_assessment["overall_quality"] <= 1.0
