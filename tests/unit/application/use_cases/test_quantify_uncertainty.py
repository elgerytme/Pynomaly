"""
Unit tests for quantify uncertainty use case.

Tests the QuantifyUncertaintyUseCase application logic including
request validation, uncertainty calculation orchestration, and response formatting.
"""

from unittest.mock import Mock

import numpy as np
import pytest

from pynomaly.application.dto.uncertainty_dto import (
    EnsembleUncertaintyRequest,
    EnsembleUncertaintyResponse,
    UncertaintyRequest,
    UncertaintyResponse,
)
from pynomaly.application.use_cases.quantify_uncertainty import (
    QuantifyUncertaintyUseCase,
)
from pynomaly.domain.entities.detection_result import DetectionResult
from pynomaly.domain.services.uncertainty_service import (
    UncertaintyQuantificationService,
)
from pynomaly.domain.value_objects.anomaly_score import AnomalyScore
from pynomaly.domain.value_objects.confidence_interval import ConfidenceInterval


class TestQuantifyUncertaintyUseCase:
    """Test cases for QuantifyUncertaintyUseCase."""

    @pytest.fixture
    def mock_uncertainty_service(self):
        """Create mock uncertainty service."""
        return Mock(spec=UncertaintyQuantificationService)

    @pytest.fixture
    def use_case(self, mock_uncertainty_service):
        """Create use case with mock service."""
        return QuantifyUncertaintyUseCase(uncertainty_service=mock_uncertainty_service)

    @pytest.fixture
    def sample_detection_results(self):
        """Create sample detection results for testing."""
        results = []
        scores = [0.1, 0.3, 0.7, 0.9, 0.2, 0.8, 0.4, 0.6, 0.5, 0.95]

        for i, score_value in enumerate(scores):
            score = AnomalyScore(value=score_value)
            result = DetectionResult(
                sample_id=f"sample_{i}",
                score=score,
                is_anomaly=score_value > 0.5,
                timestamp=None,
                model_version="test_v1",
                metadata={},
            )
            results.append(result)

        return results

    @pytest.fixture
    def sample_ensemble_results(self, sample_detection_results):
        """Create sample ensemble results for testing."""
        # Create 3 models with the same detection results structure
        return [
            sample_detection_results[:5],  # Model 1: first 5 results
            sample_detection_results[2:7],  # Model 2: middle 5 results
            sample_detection_results[5:],  # Model 3: last 5 results
        ]

    def test_execute_uncertainty_request_bootstrap(
        self, use_case, mock_uncertainty_service, sample_detection_results
    ):
        """Test executing uncertainty request with bootstrap method."""
        # Setup mock responses
        mock_uncertainty_metrics = {
            "mean_score": 0.55,
            "std_score": 0.3,
            "variance_score": 0.09,
            "coefficient_of_variation": 0.545,
            "confidence_interval_mean": ConfidenceInterval.from_bounds(
                0.4, 0.7, method="bootstrap_mean"
            ),
            "confidence_interval_std": ConfidenceInterval.from_bounds(
                0.2, 0.4, method="bootstrap_std"
            ),
            "entropy": 0.85,
        }

        mock_uncertainty_service.calculate_prediction_uncertainty.return_value = (
            mock_uncertainty_metrics
        )

        # Create request
        request = UncertaintyRequest(
            detection_results=sample_detection_results,
            method="bootstrap",
            confidence_level=0.95,
            include_prediction_intervals=True,
            include_entropy=True,
        )

        # Execute use case
        response = use_case.execute(request)

        # Verify response
        assert isinstance(response, UncertaintyResponse)
        assert response.method == "bootstrap"
        assert response.confidence_level == 0.95
        assert response.n_samples == len(sample_detection_results)
        assert len(response.confidence_intervals) > 0
        assert "mean" in response.confidence_intervals
        assert "std" in response.confidence_intervals

        # Verify service was called correctly
        mock_uncertainty_service.calculate_prediction_uncertainty.assert_called_once_with(
            detection_results=sample_detection_results, method="bootstrap"
        )

    def test_execute_uncertainty_request_normal(
        self, use_case, mock_uncertainty_service, sample_detection_results
    ):
        """Test executing uncertainty request with normal method."""
        # Setup mock responses
        mock_uncertainty_metrics = {
            "mean_score": 0.55,
            "std_score": 0.3,
            "variance_score": 0.09,
            "coefficient_of_variation": 0.545,
            "confidence_interval_mean": ConfidenceInterval.from_bounds(
                0.4, 0.7, method="normal"
            ),
            "entropy": 0.85,
        }

        mock_uncertainty_service.calculate_prediction_uncertainty.return_value = (
            mock_uncertainty_metrics
        )

        # Create request
        request = UncertaintyRequest(
            detection_results=sample_detection_results,
            method="normal",
            confidence_level=0.90,
        )

        # Execute use case
        response = use_case.execute(request)

        # Verify response
        assert response.method == "normal"
        assert response.confidence_level == 0.90
        assert "mean" in response.confidence_intervals
        assert (
            "std" not in response.confidence_intervals
        )  # Not available for normal method

    def test_execute_uncertainty_request_bayesian(
        self, use_case, mock_uncertainty_service, sample_detection_results
    ):
        """Test executing uncertainty request with Bayesian method."""
        # Setup mock responses
        mock_uncertainty_metrics = {
            "mean_score": 0.55,
            "std_score": 0.3,
            "variance_score": 0.09,
            "coefficient_of_variation": 0.545,
            "confidence_interval_mean": ConfidenceInterval.from_bounds(
                0.3, 0.8, method="bayesian"
            ),
            "entropy": 0.85,
        }

        mock_uncertainty_service.calculate_prediction_uncertainty.return_value = (
            mock_uncertainty_metrics
        )

        # Mock Bayesian CI calculation
        mock_uncertainty_service.calculate_bayesian_confidence_interval.return_value = (
            ConfidenceInterval.from_bounds(0.2, 0.7, method="bayesian_beta")
        )

        # Create request
        request = UncertaintyRequest(
            detection_results=sample_detection_results,
            method="bayesian",
            confidence_level=0.95,
        )

        # Execute use case
        response = use_case.execute(request)

        # Verify response
        assert response.method == "bayesian"
        assert "anomaly_rate" in response.confidence_intervals

    def test_execute_empty_detection_results(self, use_case):
        """Test executing with empty detection results."""
        request = UncertaintyRequest(detection_results=[])

        with pytest.raises(
            ValueError, match="Cannot quantify uncertainty without detection results"
        ):
            use_case.execute(request)

    def test_execute_ensemble_uncertainty(
        self, use_case, mock_uncertainty_service, sample_ensemble_results
    ):
        """Test executing ensemble uncertainty request."""
        # Setup mock responses
        mock_ensemble_metrics = {
            "ensemble_mean": 0.55,
            "ensemble_std": 0.1,
            "total_variance": 0.09,
            "aleatoric_uncertainty": 0.05,
            "epistemic_uncertainty": 0.04,
            "ensemble_confidence_interval": ConfidenceInterval.from_bounds(
                0.4, 0.7, method="ensemble"
            ),
        }

        mock_model_uncertainty = {
            "mean_score": 0.5,
            "std_score": 0.2,
            "confidence_interval_mean": ConfidenceInterval.from_bounds(
                0.3, 0.7, method="bootstrap"
            ),
        }

        mock_uncertainty_service.calculate_ensemble_uncertainty.return_value = (
            mock_ensemble_metrics
        )
        mock_uncertainty_service.calculate_prediction_uncertainty.return_value = (
            mock_model_uncertainty
        )

        # Create request
        request = EnsembleUncertaintyRequest(
            ensemble_results=sample_ensemble_results,
            method="bootstrap",
            confidence_level=0.95,
        )

        # Execute use case
        response = use_case.execute_ensemble_uncertainty(request)

        # Verify response
        assert isinstance(response, EnsembleUncertaintyResponse)
        assert response.method == "bootstrap"
        assert response.confidence_level == 0.95
        assert response.n_models == 3
        assert len(response.model_uncertainties) == 3
        assert "disagreement_rate" in response.disagreement_metrics
        assert "avg_pairwise_disagreement" in response.disagreement_metrics
        assert "ensemble_entropy" in response.disagreement_metrics

        # Verify service calls
        mock_uncertainty_service.calculate_ensemble_uncertainty.assert_called_once()
        assert mock_uncertainty_service.calculate_prediction_uncertainty.call_count == 3

    def test_execute_ensemble_uncertainty_empty_results(self, use_case):
        """Test executing ensemble uncertainty with empty results."""
        request = EnsembleUncertaintyRequest(ensemble_results=[])

        with pytest.raises(
            ValueError, match="Cannot quantify ensemble uncertainty without results"
        ):
            use_case.execute_ensemble_uncertainty(request)

    def test_calculate_bootstrap_interval(self, use_case, mock_uncertainty_service):
        """Test calculate_bootstrap_interval method."""
        scores = [0.1, 0.2, 0.3, 0.4, 0.5]
        expected_ci = ConfidenceInterval.from_bounds(0.2, 0.4, method="bootstrap_mean")

        mock_uncertainty_service.calculate_bootstrap_confidence_interval.return_value = expected_ci

        result = use_case.calculate_bootstrap_interval(
            scores=scores, confidence_level=0.90, n_bootstrap=500, statistic="mean"
        )

        assert result == expected_ci
        mock_uncertainty_service.calculate_bootstrap_confidence_interval.assert_called_once_with(
            scores=scores,
            confidence_level=0.90,
            n_bootstrap=500,
            statistic_function="mean",
        )

    def test_calculate_bayesian_interval(self, use_case, mock_uncertainty_service):
        """Test calculate_bayesian_interval method."""
        binary_scores = [0, 1, 0, 1, 1]
        expected_ci = ConfidenceInterval.from_bounds(0.3, 0.8, method="bayesian_beta")

        mock_uncertainty_service.calculate_bayesian_confidence_interval.return_value = (
            expected_ci
        )

        result = use_case.calculate_bayesian_interval(
            binary_scores=binary_scores,
            confidence_level=0.95,
            prior_alpha=2.0,
            prior_beta=2.0,
        )

        assert result == expected_ci
        mock_uncertainty_service.calculate_bayesian_confidence_interval.assert_called_once_with(
            scores=binary_scores, confidence_level=0.95, prior_alpha=2.0, prior_beta=2.0
        )

    def test_calculate_prediction_interval(self, use_case, mock_uncertainty_service):
        """Test calculate_prediction_interval method."""
        training_scores = [0.1, 0.2, 0.3, 0.4, 0.5]
        expected_ci = ConfidenceInterval.from_bounds(
            -0.1, 0.9, method="prediction_interval"
        )

        mock_uncertainty_service._calculate_prediction_interval.return_value = (
            expected_ci
        )

        result = use_case.calculate_prediction_interval(
            training_scores=training_scores, confidence_level=0.95
        )

        assert result == expected_ci
        mock_uncertainty_service._calculate_prediction_interval.assert_called_once_with(
            scores=np.array(training_scores), confidence_level=0.95
        )

    def test_calculate_confidence_intervals_bootstrap(
        self, use_case, mock_uncertainty_service, sample_detection_results
    ):
        """Test _calculate_confidence_intervals method with bootstrap."""
        mock_ci_mean = ConfidenceInterval.from_bounds(0.4, 0.6, method="bootstrap_mean")
        mock_ci_std = ConfidenceInterval.from_bounds(0.2, 0.4, method="bootstrap_std")

        mock_uncertainty_service.calculate_bootstrap_confidence_interval.side_effect = [
            mock_ci_mean,
            mock_ci_std,
        ]

        result = use_case._calculate_confidence_intervals(
            detection_results=sample_detection_results,
            confidence_level=0.95,
            method="bootstrap",
        )

        assert "mean" in result
        assert "std" in result
        assert result["mean"] == mock_ci_mean
        assert result["std"] == mock_ci_std

    def test_calculate_confidence_intervals_normal(
        self, use_case, mock_uncertainty_service, sample_detection_results
    ):
        """Test _calculate_confidence_intervals method with normal."""
        mock_ci_mean = ConfidenceInterval.from_bounds(0.4, 0.6, method="normal")

        mock_uncertainty_service.calculate_normal_confidence_interval.return_value = (
            mock_ci_mean
        )

        result = use_case._calculate_confidence_intervals(
            detection_results=sample_detection_results,
            confidence_level=0.95,
            method="normal",
        )

        assert "mean" in result
        assert "std" not in result
        assert result["mean"] == mock_ci_mean

    def test_calculate_additional_metrics(
        self, use_case, mock_uncertainty_service, sample_detection_results
    ):
        """Test _calculate_additional_metrics method."""
        mock_prediction_interval = ConfidenceInterval.from_bounds(
            0.0, 1.0, method="prediction_interval"
        )
        mock_uncertainty_service._calculate_prediction_interval.return_value = (
            mock_prediction_interval
        )
        mock_uncertainty_service._calculate_entropy.return_value = 0.75

        result = use_case._calculate_additional_metrics(
            detection_results=sample_detection_results,
            include_prediction_intervals=True,
            include_entropy=True,
        )

        assert "prediction_interval" in result
        assert "entropy" in result
        assert "range" in result
        assert "iqr" in result
        assert "median_absolute_deviation" in result

        assert result["prediction_interval"] == mock_prediction_interval
        assert result["entropy"] == 0.75

    def test_calculate_ensemble_disagreement(self, use_case, sample_ensemble_results):
        """Test _calculate_ensemble_disagreement method."""
        # Create ensemble results with known disagreements
        ensemble_results = []

        # Model 1: [True, False, True, False, True]
        model1_results = []
        for i, is_anomaly in enumerate([True, False, True, False, True]):
            score = AnomalyScore(value=0.8 if is_anomaly else 0.2)
            result = DetectionResult(
                sample_id=f"sample_{i}",
                score=score,
                is_anomaly=is_anomaly,
                timestamp=None,
                model_version="model1",
                metadata={},
            )
            model1_results.append(result)

        # Model 2: [True, True, False, False, True]
        model2_results = []
        for i, is_anomaly in enumerate([True, True, False, False, True]):
            score = AnomalyScore(value=0.8 if is_anomaly else 0.2)
            result = DetectionResult(
                sample_id=f"sample_{i}",
                score=score,
                is_anomaly=is_anomaly,
                timestamp=None,
                model_version="model2",
                metadata={},
            )
            model2_results.append(result)

        ensemble_results = [model1_results, model2_results]

        disagreement_metrics = use_case._calculate_ensemble_disagreement(
            ensemble_results
        )

        assert "disagreement_rate" in disagreement_metrics
        assert "avg_pairwise_disagreement" in disagreement_metrics
        assert "ensemble_entropy" in disagreement_metrics

        # Expected disagreements: positions 1 and 2 (2 out of 5 = 0.4)
        assert disagreement_metrics["disagreement_rate"] == 0.4
        assert disagreement_metrics["avg_pairwise_disagreement"] == 0.4

    def test_uncertainty_response_to_dict(self, sample_detection_results):
        """Test UncertaintyResponse to_dict method."""
        confidence_intervals = {
            "mean": ConfidenceInterval.from_bounds(0.4, 0.6, method="bootstrap"),
            "std": ConfidenceInterval.from_bounds(0.2, 0.4, method="bootstrap"),
        }

        uncertainty_metrics = {
            "mean_score": 0.5,
            "std_score": 0.3,
            "confidence_interval_mean": ConfidenceInterval.from_bounds(
                0.4, 0.6, method="bootstrap"
            ),
        }

        additional_metrics = {"entropy": 0.75, "range": 0.8}

        response = UncertaintyResponse(
            confidence_intervals=confidence_intervals,
            uncertainty_metrics=uncertainty_metrics,
            additional_metrics=additional_metrics,
            method="bootstrap",
            confidence_level=0.95,
            n_samples=10,
        )

        result_dict = response.to_dict()

        assert "confidence_intervals" in result_dict
        assert "uncertainty_metrics" in result_dict
        assert "additional_metrics" in result_dict
        assert result_dict["method"] == "bootstrap"
        assert result_dict["confidence_level"] == 0.95
        assert result_dict["n_samples"] == 10

        # Check that ConfidenceInterval objects are converted to dicts
        assert isinstance(result_dict["confidence_intervals"]["mean"], dict)
        assert isinstance(
            result_dict["uncertainty_metrics"]["confidence_interval_mean"], dict
        )

    def test_uncertainty_response_get_summary(self):
        """Test UncertaintyResponse get_summary method."""
        confidence_intervals = {
            "mean": ConfidenceInterval.from_bounds(0.4, 0.6, method="bootstrap")
        }

        uncertainty_metrics = {
            "mean_score": 0.5,
            "std_score": 0.3,
            "coefficient_of_variation": 0.6,
        }

        additional_metrics = {"entropy": 0.75}

        response = UncertaintyResponse(
            confidence_intervals=confidence_intervals,
            uncertainty_metrics=uncertainty_metrics,
            additional_metrics=additional_metrics,
            method="bootstrap",
            confidence_level=0.95,
            n_samples=10,
        )

        summary = response.get_summary()

        assert summary["method"] == "bootstrap"
        assert summary["confidence_level"] == 0.95
        assert summary["n_samples"] == 10
        assert summary["mean_score"] == 0.5
        assert summary["std_score"] == 0.3
        assert summary["coefficient_of_variation"] == 0.6
        assert summary["entropy"] == 0.75
        assert "mean_ci_width" in summary
