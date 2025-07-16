"""Tests for Uncertainty DTOs."""

from uuid import uuid4

import pytest

from monorepo.application.dto.uncertainty_dto import (
    BayesianRequest,
    BootstrapRequest,
    EnsembleUncertaintyRequest,
    EnsembleUncertaintyResponse,
    PredictionIntervalRequest,
    UncertaintyRequest,
    UncertaintyResponse,
)
from monorepo.domain.entities.detection_result import DetectionResult
from monorepo.domain.value_objects.anomaly_score import AnomalyScore
from monorepo.domain.value_objects.confidence_interval import ConfidenceInterval


class TestUncertaintyRequest:
    """Test suite for UncertaintyRequest."""

    def test_valid_creation(self):
        """Test creating a valid uncertainty request."""
        detection_results = [
            DetectionResult(
                id=str(uuid4()),
                detector_id=str(uuid4()),
                dataset_id=str(uuid4()),
                anomaly_score=AnomalyScore(0.8),
                is_anomaly=True,
                sample_index=i,
            )
            for i in range(5)
        ]

        request = UncertaintyRequest(
            detection_results=detection_results,
            method="bootstrap",
            confidence_level=0.95,
            include_prediction_intervals=True,
            include_entropy=True,
            n_bootstrap=2000,
        )

        assert len(request.detection_results) == 5
        assert request.method == "bootstrap"
        assert request.confidence_level == 0.95
        assert request.include_prediction_intervals is True
        assert request.include_entropy is True
        assert request.n_bootstrap == 2000

    def test_default_values(self):
        """Test default values."""
        detection_results = [
            DetectionResult(
                id=str(uuid4()),
                detector_id=str(uuid4()),
                dataset_id=str(uuid4()),
                anomaly_score=AnomalyScore(0.7),
                is_anomaly=False,
                sample_index=0,
            )
        ]

        request = UncertaintyRequest(detection_results=detection_results)

        assert len(request.detection_results) == 1
        assert request.method == "bootstrap"
        assert request.confidence_level == 0.95
        assert request.include_prediction_intervals is True
        assert request.include_entropy is True
        assert request.n_bootstrap == 1000

    def test_empty_detection_results_validation(self):
        """Test validation for empty detection results."""
        with pytest.raises(ValueError, match="Detection results cannot be empty"):
            UncertaintyRequest(detection_results=[])

    def test_invalid_method_validation(self):
        """Test validation for invalid method."""
        detection_results = [
            DetectionResult(
                id=str(uuid4()),
                detector_id=str(uuid4()),
                dataset_id=str(uuid4()),
                anomaly_score=AnomalyScore(0.7),
                is_anomaly=False,
                sample_index=0,
            )
        ]

        with pytest.raises(ValueError, match="Unknown method: invalid_method"):
            UncertaintyRequest(
                detection_results=detection_results, method="invalid_method"
            )

    def test_valid_methods(self):
        """Test all valid methods."""
        detection_results = [
            DetectionResult(
                id=str(uuid4()),
                detector_id=str(uuid4()),
                dataset_id=str(uuid4()),
                anomaly_score=AnomalyScore(0.7),
                is_anomaly=False,
                sample_index=0,
            )
        ]

        valid_methods = ["bootstrap", "normal", "bayesian"]
        for method in valid_methods:
            request = UncertaintyRequest(
                detection_results=detection_results, method=method
            )
            assert request.method == method

    def test_invalid_confidence_level_validation(self):
        """Test validation for invalid confidence level."""
        detection_results = [
            DetectionResult(
                id=str(uuid4()),
                detector_id=str(uuid4()),
                dataset_id=str(uuid4()),
                anomaly_score=AnomalyScore(0.7),
                is_anomaly=False,
                sample_index=0,
            )
        ]

        # Too low
        with pytest.raises(
            ValueError, match="Confidence level must be between 0 and 1"
        ):
            UncertaintyRequest(
                detection_results=detection_results, confidence_level=-0.1
            )

        # Too high
        with pytest.raises(
            ValueError, match="Confidence level must be between 0 and 1"
        ):
            UncertaintyRequest(
                detection_results=detection_results, confidence_level=1.1
            )

    def test_boundary_confidence_levels(self):
        """Test boundary confidence levels."""
        detection_results = [
            DetectionResult(
                id=str(uuid4()),
                detector_id=str(uuid4()),
                dataset_id=str(uuid4()),
                anomaly_score=AnomalyScore(0.7),
                is_anomaly=False,
                sample_index=0,
            )
        ]

        # Valid boundary values
        for confidence_level in [0.0, 0.5, 0.9, 0.99, 1.0]:
            request = UncertaintyRequest(
                detection_results=detection_results, confidence_level=confidence_level
            )
            assert request.confidence_level == confidence_level

    def test_invalid_n_bootstrap_validation(self):
        """Test validation for invalid n_bootstrap."""
        detection_results = [
            DetectionResult(
                id=str(uuid4()),
                detector_id=str(uuid4()),
                dataset_id=str(uuid4()),
                anomaly_score=AnomalyScore(0.7),
                is_anomaly=False,
                sample_index=0,
            )
        ]

        with pytest.raises(ValueError, match="n_bootstrap must be at least 100"):
            UncertaintyRequest(detection_results=detection_results, n_bootstrap=50)

    def test_minimum_n_bootstrap(self):
        """Test minimum n_bootstrap value."""
        detection_results = [
            DetectionResult(
                id=str(uuid4()),
                detector_id=str(uuid4()),
                dataset_id=str(uuid4()),
                anomaly_score=AnomalyScore(0.7),
                is_anomaly=False,
                sample_index=0,
            )
        ]

        request = UncertaintyRequest(
            detection_results=detection_results, n_bootstrap=100
        )
        assert request.n_bootstrap == 100


class TestUncertaintyResponse:
    """Test suite for UncertaintyResponse."""

    def test_valid_creation(self):
        """Test creating a valid uncertainty response."""
        confidence_intervals = {
            "mean_score": ConfidenceInterval(
                lower=0.6, upper=0.8, confidence_level=0.95
            ),
            "std_score": ConfidenceInterval(
                lower=0.1, upper=0.3, confidence_level=0.95
            ),
        }

        uncertainty_metrics = {
            "mean_score": 0.7,
            "std_score": 0.2,
            "coefficient_of_variation": 0.286,
        }

        additional_metrics = {"entropy": 0.45, "variance": 0.04, "skewness": 0.1}

        response = UncertaintyResponse(
            confidence_intervals=confidence_intervals,
            uncertainty_metrics=uncertainty_metrics,
            additional_metrics=additional_metrics,
            method="bootstrap",
            confidence_level=0.95,
            n_samples=1000,
        )

        assert len(response.confidence_intervals) == 2
        assert len(response.uncertainty_metrics) == 3
        assert len(response.additional_metrics) == 3
        assert response.method == "bootstrap"
        assert response.confidence_level == 0.95
        assert response.n_samples == 1000

    def test_to_dict_method(self):
        """Test to_dict method."""
        confidence_intervals = {
            "mean_score": ConfidenceInterval(
                lower=0.6, upper=0.8, confidence_level=0.95
            )
        }

        uncertainty_metrics = {
            "mean_score": 0.7,
            "ci_metric": ConfidenceInterval(
                lower=0.1, upper=0.3, confidence_level=0.95
            ),
        }

        additional_metrics = {"entropy": 0.45}

        response = UncertaintyResponse(
            confidence_intervals=confidence_intervals,
            uncertainty_metrics=uncertainty_metrics,
            additional_metrics=additional_metrics,
            method="normal",
            confidence_level=0.95,
            n_samples=500,
        )

        result_dict = response.to_dict()

        assert "confidence_intervals" in result_dict
        assert "uncertainty_metrics" in result_dict
        assert "additional_metrics" in result_dict
        assert result_dict["method"] == "normal"
        assert result_dict["confidence_level"] == 0.95
        assert result_dict["n_samples"] == 500

        # Check that confidence intervals are converted to dicts
        assert isinstance(result_dict["confidence_intervals"]["mean_score"], dict)
        assert isinstance(result_dict["uncertainty_metrics"]["ci_metric"], dict)
        assert result_dict["uncertainty_metrics"]["mean_score"] == 0.7
        assert result_dict["additional_metrics"]["entropy"] == 0.45

    def test_get_summary_method(self):
        """Test get_summary method."""
        confidence_intervals = {
            "mean_score": ConfidenceInterval(
                lower=0.6, upper=0.8, confidence_level=0.95
            ),
            "std_score": ConfidenceInterval(
                lower=0.1, upper=0.3, confidence_level=0.95
            ),
        }

        uncertainty_metrics = {
            "mean_score": 0.7,
            "std_score": 0.2,
            "coefficient_of_variation": 0.286,
        }

        additional_metrics = {"entropy": 0.45, "variance": 0.04}

        response = UncertaintyResponse(
            confidence_intervals=confidence_intervals,
            uncertainty_metrics=uncertainty_metrics,
            additional_metrics=additional_metrics,
            method="bootstrap",
            confidence_level=0.95,
            n_samples=1000,
        )

        summary = response.get_summary()

        assert summary["method"] == "bootstrap"
        assert summary["confidence_level"] == 0.95
        assert summary["n_samples"] == 1000
        assert summary["mean_score"] == 0.7
        assert summary["std_score"] == 0.2
        assert summary["coefficient_of_variation"] == 0.286
        assert summary["entropy"] == 0.45
        assert "mean_score_ci_width" in summary
        assert "std_score_ci_width" in summary

    def test_get_summary_with_missing_metrics(self):
        """Test get_summary method with missing metrics."""
        confidence_intervals = {}
        uncertainty_metrics = {}
        additional_metrics = {}

        response = UncertaintyResponse(
            confidence_intervals=confidence_intervals,
            uncertainty_metrics=uncertainty_metrics,
            additional_metrics=additional_metrics,
            method="bayesian",
            confidence_level=0.99,
            n_samples=200,
        )

        summary = response.get_summary()

        assert summary["method"] == "bayesian"
        assert summary["confidence_level"] == 0.99
        assert summary["n_samples"] == 200
        assert "mean_score" not in summary
        assert "std_score" not in summary
        assert "entropy" not in summary


class TestEnsembleUncertaintyRequest:
    """Test suite for EnsembleUncertaintyRequest."""

    def test_valid_creation(self):
        """Test creating a valid ensemble uncertainty request."""
        # Create ensemble results (3 models, 5 samples each)
        ensemble_results = []
        for model_idx in range(3):
            model_results = []
            for sample_idx in range(5):
                result = DetectionResult(
                    id=str(uuid4()),
                    detector_id=f"detector_{model_idx}",
                    dataset_id=str(uuid4()),
                    anomaly_score=AnomalyScore(
                        0.5 + model_idx * 0.1 + sample_idx * 0.05
                    ),
                    is_anomaly=False,
                    sample_index=sample_idx,
                )
                model_results.append(result)
            ensemble_results.append(model_results)

        request = EnsembleUncertaintyRequest(
            ensemble_results=ensemble_results,
            method="bootstrap",
            confidence_level=0.95,
            include_disagreement=True,
        )

        assert len(request.ensemble_results) == 3
        assert len(request.ensemble_results[0]) == 5
        assert request.method == "bootstrap"
        assert request.confidence_level == 0.95
        assert request.include_disagreement is True

    def test_default_values(self):
        """Test default values."""
        ensemble_results = [
            [
                DetectionResult(
                    id=str(uuid4()),
                    detector_id="detector_0",
                    dataset_id=str(uuid4()),
                    anomaly_score=AnomalyScore(0.7),
                    is_anomaly=False,
                    sample_index=0,
                )
            ],
            [
                DetectionResult(
                    id=str(uuid4()),
                    detector_id="detector_1",
                    dataset_id=str(uuid4()),
                    anomaly_score=AnomalyScore(0.8),
                    is_anomaly=True,
                    sample_index=0,
                )
            ],
        ]

        request = EnsembleUncertaintyRequest(ensemble_results=ensemble_results)

        assert len(request.ensemble_results) == 2
        assert request.method == "bootstrap"
        assert request.confidence_level == 0.95
        assert request.include_disagreement is True

    def test_empty_ensemble_results_validation(self):
        """Test validation for empty ensemble results."""
        with pytest.raises(ValueError, match="Ensemble results cannot be empty"):
            EnsembleUncertaintyRequest(ensemble_results=[])

    def test_insufficient_models_validation(self):
        """Test validation for insufficient models."""
        ensemble_results = [
            [
                DetectionResult(
                    id=str(uuid4()),
                    detector_id="detector_0",
                    dataset_id=str(uuid4()),
                    anomaly_score=AnomalyScore(0.7),
                    is_anomaly=False,
                    sample_index=0,
                )
            ]
        ]

        with pytest.raises(ValueError, match="Ensemble must contain at least 2 models"):
            EnsembleUncertaintyRequest(ensemble_results=ensemble_results)

    def test_mismatched_result_counts_validation(self):
        """Test validation for mismatched result counts."""
        ensemble_results = [
            [
                DetectionResult(
                    id=str(uuid4()),
                    detector_id="detector_0",
                    dataset_id=str(uuid4()),
                    anomaly_score=AnomalyScore(0.7),
                    is_anomaly=False,
                    sample_index=0,
                ),
                DetectionResult(
                    id=str(uuid4()),
                    detector_id="detector_0",
                    dataset_id=str(uuid4()),
                    anomaly_score=AnomalyScore(0.8),
                    is_anomaly=True,
                    sample_index=1,
                ),
            ],
            [
                DetectionResult(
                    id=str(uuid4()),
                    detector_id="detector_1",
                    dataset_id=str(uuid4()),
                    anomaly_score=AnomalyScore(0.6),
                    is_anomaly=False,
                    sample_index=0,
                )
                # Missing second result
            ],
        ]

        with pytest.raises(
            ValueError,
            match="All models must have the same number of detection results",
        ):
            EnsembleUncertaintyRequest(ensemble_results=ensemble_results)

    def test_valid_methods(self):
        """Test all valid methods."""
        ensemble_results = [
            [
                DetectionResult(
                    id=str(uuid4()),
                    detector_id="detector_0",
                    dataset_id=str(uuid4()),
                    anomaly_score=AnomalyScore(0.7),
                    is_anomaly=False,
                    sample_index=0,
                )
            ],
            [
                DetectionResult(
                    id=str(uuid4()),
                    detector_id="detector_1",
                    dataset_id=str(uuid4()),
                    anomaly_score=AnomalyScore(0.8),
                    is_anomaly=True,
                    sample_index=0,
                )
            ],
        ]

        valid_methods = ["bootstrap", "normal", "bayesian"]
        for method in valid_methods:
            request = EnsembleUncertaintyRequest(
                ensemble_results=ensemble_results, method=method
            )
            assert request.method == method

    def test_invalid_method_validation(self):
        """Test validation for invalid method."""
        ensemble_results = [
            [
                DetectionResult(
                    id=str(uuid4()),
                    detector_id="detector_0",
                    dataset_id=str(uuid4()),
                    anomaly_score=AnomalyScore(0.7),
                    is_anomaly=False,
                    sample_index=0,
                )
            ],
            [
                DetectionResult(
                    id=str(uuid4()),
                    detector_id="detector_1",
                    dataset_id=str(uuid4()),
                    anomaly_score=AnomalyScore(0.8),
                    is_anomaly=True,
                    sample_index=0,
                )
            ],
        ]

        with pytest.raises(ValueError, match="Unknown method: invalid_method"):
            EnsembleUncertaintyRequest(
                ensemble_results=ensemble_results, method="invalid_method"
            )

    def test_invalid_confidence_level_validation(self):
        """Test validation for invalid confidence level."""
        ensemble_results = [
            [
                DetectionResult(
                    id=str(uuid4()),
                    detector_id="detector_0",
                    dataset_id=str(uuid4()),
                    anomaly_score=AnomalyScore(0.7),
                    is_anomaly=False,
                    sample_index=0,
                )
            ],
            [
                DetectionResult(
                    id=str(uuid4()),
                    detector_id="detector_1",
                    dataset_id=str(uuid4()),
                    anomaly_score=AnomalyScore(0.8),
                    is_anomaly=True,
                    sample_index=0,
                )
            ],
        ]

        # Too low
        with pytest.raises(
            ValueError, match="Confidence level must be between 0 and 1"
        ):
            EnsembleUncertaintyRequest(
                ensemble_results=ensemble_results, confidence_level=-0.1
            )

        # Too high
        with pytest.raises(
            ValueError, match="Confidence level must be between 0 and 1"
        ):
            EnsembleUncertaintyRequest(
                ensemble_results=ensemble_results, confidence_level=1.1
            )


class TestEnsembleUncertaintyResponse:
    """Test suite for EnsembleUncertaintyResponse."""

    def test_valid_creation(self):
        """Test creating a valid ensemble uncertainty response."""
        ensemble_metrics = {
            "ensemble_mean": 0.75,
            "ensemble_std": 0.15,
            "aleatoric_uncertainty": 0.1,
            "epistemic_uncertainty": 0.05,
        }

        model_uncertainties = [
            {"model_id": 0, "metrics": {"mean": 0.7, "std": 0.12}},
            {"model_id": 1, "metrics": {"mean": 0.8, "std": 0.18}},
            {"model_id": 2, "metrics": {"mean": 0.75, "std": 0.15}},
        ]

        disagreement_metrics = {
            "variance": 0.02,
            "range": 0.1,
            "disagreement_rate": 0.3,
        }

        response = EnsembleUncertaintyResponse(
            ensemble_metrics=ensemble_metrics,
            model_uncertainties=model_uncertainties,
            disagreement_metrics=disagreement_metrics,
            method="bootstrap",
            confidence_level=0.95,
            n_models=3,
        )

        assert len(response.ensemble_metrics) == 4
        assert len(response.model_uncertainties) == 3
        assert len(response.disagreement_metrics) == 3
        assert response.method == "bootstrap"
        assert response.confidence_level == 0.95
        assert response.n_models == 3

    def test_to_dict_method(self):
        """Test to_dict method."""
        ensemble_metrics = {
            "ensemble_mean": 0.75,
            "ensemble_ci": ConfidenceInterval(
                lower=0.6, upper=0.9, confidence_level=0.95
            ),
        }

        model_uncertainties = [{"model_id": 0, "metrics": {"mean": 0.7}}]

        disagreement_metrics = {"variance": 0.02}

        response = EnsembleUncertaintyResponse(
            ensemble_metrics=ensemble_metrics,
            model_uncertainties=model_uncertainties,
            disagreement_metrics=disagreement_metrics,
            method="normal",
            confidence_level=0.95,
            n_models=1,
        )

        result_dict = response.to_dict()

        assert "ensemble_metrics" in result_dict
        assert "model_uncertainties" in result_dict
        assert "disagreement_metrics" in result_dict
        assert result_dict["method"] == "normal"
        assert result_dict["confidence_level"] == 0.95
        assert result_dict["n_models"] == 1

        # Check that confidence intervals are converted to dicts
        assert isinstance(result_dict["ensemble_metrics"]["ensemble_ci"], dict)
        assert result_dict["ensemble_metrics"]["ensemble_mean"] == 0.75

    def test_get_summary_method(self):
        """Test get_summary method."""
        ensemble_metrics = {
            "ensemble_mean": 0.75,
            "ensemble_std": 0.15,
            "aleatoric_uncertainty": 0.1,
            "epistemic_uncertainty": 0.05,
        }

        model_uncertainties = [{"model_id": 0, "metrics": {"mean": 0.7}}]

        disagreement_metrics = {
            "variance": 0.02,
            "range": 0.1,
            "disagreement_rate": 0.3,
        }

        response = EnsembleUncertaintyResponse(
            ensemble_metrics=ensemble_metrics,
            model_uncertainties=model_uncertainties,
            disagreement_metrics=disagreement_metrics,
            method="bayesian",
            confidence_level=0.99,
            n_models=5,
        )

        summary = response.get_summary()

        assert summary["method"] == "bayesian"
        assert summary["confidence_level"] == 0.99
        assert summary["n_models"] == 5
        assert summary["ensemble_mean"] == 0.75
        assert summary["ensemble_std"] == 0.15
        assert summary["aleatoric_uncertainty"] == 0.1
        assert summary["epistemic_uncertainty"] == 0.05
        assert summary["variance"] == 0.02
        assert summary["range"] == 0.1
        assert summary["disagreement_rate"] == 0.3

    def test_get_summary_with_missing_metrics(self):
        """Test get_summary method with missing metrics."""
        ensemble_metrics = {}
        model_uncertainties = []
        disagreement_metrics = {}

        response = EnsembleUncertaintyResponse(
            ensemble_metrics=ensemble_metrics,
            model_uncertainties=model_uncertainties,
            disagreement_metrics=disagreement_metrics,
            method="bootstrap",
            confidence_level=0.95,
            n_models=2,
        )

        summary = response.get_summary()

        assert summary["method"] == "bootstrap"
        assert summary["confidence_level"] == 0.95
        assert summary["n_models"] == 2
        assert "ensemble_mean" not in summary
        assert "aleatoric_uncertainty" not in summary


class TestBootstrapRequest:
    """Test suite for BootstrapRequest."""

    def test_valid_creation(self):
        """Test creating a valid bootstrap request."""
        scores = [0.1, 0.3, 0.5, 0.7, 0.9, 0.4, 0.6, 0.8, 0.2]

        request = BootstrapRequest(
            scores=scores, confidence_level=0.95, n_bootstrap=2000, statistic="mean"
        )

        assert request.scores == scores
        assert request.confidence_level == 0.95
        assert request.n_bootstrap == 2000
        assert request.statistic == "mean"

    def test_default_values(self):
        """Test default values."""
        scores = [0.5, 0.6, 0.7]

        request = BootstrapRequest(scores=scores)

        assert request.scores == scores
        assert request.confidence_level == 0.95
        assert request.n_bootstrap == 1000
        assert request.statistic == "mean"

    def test_empty_scores_validation(self):
        """Test validation for empty scores."""
        with pytest.raises(ValueError, match="Scores cannot be empty"):
            BootstrapRequest(scores=[])

    def test_invalid_confidence_level_validation(self):
        """Test validation for invalid confidence level."""
        scores = [0.5, 0.6, 0.7]

        # Too low
        with pytest.raises(
            ValueError, match="Confidence level must be between 0 and 1"
        ):
            BootstrapRequest(scores=scores, confidence_level=-0.1)

        # Too high
        with pytest.raises(
            ValueError, match="Confidence level must be between 0 and 1"
        ):
            BootstrapRequest(scores=scores, confidence_level=1.1)

    def test_invalid_n_bootstrap_validation(self):
        """Test validation for invalid n_bootstrap."""
        scores = [0.5, 0.6, 0.7]

        with pytest.raises(ValueError, match="n_bootstrap must be at least 100"):
            BootstrapRequest(scores=scores, n_bootstrap=50)

    def test_valid_statistics(self):
        """Test all valid statistics."""
        scores = [0.5, 0.6, 0.7]
        valid_statistics = ["mean", "median", "std", "var"]

        for statistic in valid_statistics:
            request = BootstrapRequest(scores=scores, statistic=statistic)
            assert request.statistic == statistic

    def test_invalid_statistic_validation(self):
        """Test validation for invalid statistic."""
        scores = [0.5, 0.6, 0.7]

        with pytest.raises(ValueError, match="Unknown statistic: invalid_stat"):
            BootstrapRequest(scores=scores, statistic="invalid_stat")

    def test_minimum_n_bootstrap(self):
        """Test minimum n_bootstrap value."""
        scores = [0.5, 0.6, 0.7]

        request = BootstrapRequest(scores=scores, n_bootstrap=100)
        assert request.n_bootstrap == 100

    def test_boundary_confidence_levels(self):
        """Test boundary confidence levels."""
        scores = [0.5, 0.6, 0.7]

        # Valid boundary values
        for confidence_level in [0.0, 0.5, 0.9, 0.99, 1.0]:
            request = BootstrapRequest(scores=scores, confidence_level=confidence_level)
            assert request.confidence_level == confidence_level


class TestBayesianRequest:
    """Test suite for BayesianRequest."""

    def test_valid_creation(self):
        """Test creating a valid Bayesian request."""
        binary_scores = [0, 1, 0, 0, 1, 1, 0, 1, 0, 1]

        request = BayesianRequest(
            binary_scores=binary_scores,
            confidence_level=0.95,
            prior_alpha=2.0,
            prior_beta=3.0,
        )

        assert request.binary_scores == binary_scores
        assert request.confidence_level == 0.95
        assert request.prior_alpha == 2.0
        assert request.prior_beta == 3.0

    def test_default_values(self):
        """Test default values."""
        binary_scores = [0, 1, 1, 0]

        request = BayesianRequest(binary_scores=binary_scores)

        assert request.binary_scores == binary_scores
        assert request.confidence_level == 0.95
        assert request.prior_alpha == 1.0
        assert request.prior_beta == 1.0

    def test_empty_binary_scores_validation(self):
        """Test validation for empty binary scores."""
        with pytest.raises(ValueError, match="Binary scores cannot be empty"):
            BayesianRequest(binary_scores=[])

    def test_non_binary_scores_validation(self):
        """Test validation for non-binary scores."""
        # Contains value other than 0 or 1
        with pytest.raises(ValueError, match="Binary scores must contain only 0 and 1"):
            BayesianRequest(binary_scores=[0, 1, 2])

        # Contains negative value
        with pytest.raises(ValueError, match="Binary scores must contain only 0 and 1"):
            BayesianRequest(binary_scores=[0, 1, -1])

        # Contains float value
        with pytest.raises(ValueError, match="Binary scores must contain only 0 and 1"):
            BayesianRequest(binary_scores=[0, 1, 0.5])

    def test_valid_binary_scores(self):
        """Test valid binary scores."""
        valid_binary_scores = [[0, 1], [1, 1, 1], [0, 0, 0], [0, 1, 0, 1, 0, 1]]

        for binary_scores in valid_binary_scores:
            request = BayesianRequest(binary_scores=binary_scores)
            assert request.binary_scores == binary_scores

    def test_invalid_confidence_level_validation(self):
        """Test validation for invalid confidence level."""
        binary_scores = [0, 1, 0, 1]

        # Too low
        with pytest.raises(
            ValueError, match="Confidence level must be between 0 and 1"
        ):
            BayesianRequest(binary_scores=binary_scores, confidence_level=-0.1)

        # Too high
        with pytest.raises(
            ValueError, match="Confidence level must be between 0 and 1"
        ):
            BayesianRequest(binary_scores=binary_scores, confidence_level=1.1)

    def test_invalid_prior_parameters_validation(self):
        """Test validation for invalid prior parameters."""
        binary_scores = [0, 1, 0, 1]

        # Zero alpha
        with pytest.raises(ValueError, match="Prior parameters must be positive"):
            BayesianRequest(binary_scores=binary_scores, prior_alpha=0.0)

        # Negative alpha
        with pytest.raises(ValueError, match="Prior parameters must be positive"):
            BayesianRequest(binary_scores=binary_scores, prior_alpha=-1.0)

        # Zero beta
        with pytest.raises(ValueError, match="Prior parameters must be positive"):
            BayesianRequest(binary_scores=binary_scores, prior_beta=0.0)

        # Negative beta
        with pytest.raises(ValueError, match="Prior parameters must be positive"):
            BayesianRequest(binary_scores=binary_scores, prior_beta=-1.0)

    def test_valid_prior_parameters(self):
        """Test valid prior parameters."""
        binary_scores = [0, 1, 0, 1]

        # Small positive values
        request = BayesianRequest(
            binary_scores=binary_scores, prior_alpha=0.1, prior_beta=0.1
        )
        assert request.prior_alpha == 0.1
        assert request.prior_beta == 0.1

        # Large positive values
        request = BayesianRequest(
            binary_scores=binary_scores, prior_alpha=10.0, prior_beta=5.0
        )
        assert request.prior_alpha == 10.0
        assert request.prior_beta == 5.0


class TestPredictionIntervalRequest:
    """Test suite for PredictionIntervalRequest."""

    def test_valid_creation(self):
        """Test creating a valid prediction interval request."""
        training_scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.35, 0.45]

        request = PredictionIntervalRequest(
            training_scores=training_scores, confidence_level=0.95
        )

        assert request.training_scores == training_scores
        assert request.confidence_level == 0.95

    def test_default_values(self):
        """Test default values."""
        training_scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        request = PredictionIntervalRequest(training_scores=training_scores)

        assert request.training_scores == training_scores
        assert request.confidence_level == 0.95

    def test_empty_training_scores_validation(self):
        """Test validation for empty training scores."""
        with pytest.raises(ValueError, match="Training scores cannot be empty"):
            PredictionIntervalRequest(training_scores=[])

    def test_insufficient_training_scores_validation(self):
        """Test validation for insufficient training scores."""
        # Only 5 scores (need at least 10)
        training_scores = [0.1, 0.2, 0.3, 0.4, 0.5]

        with pytest.raises(ValueError, match="Need at least 10 training scores"):
            PredictionIntervalRequest(training_scores=training_scores)

    def test_minimum_training_scores(self):
        """Test minimum training scores requirement."""
        # Exactly 10 scores (minimum required)
        training_scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        request = PredictionIntervalRequest(training_scores=training_scores)
        assert len(request.training_scores) == 10

    def test_invalid_confidence_level_validation(self):
        """Test validation for invalid confidence level."""
        training_scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        # Too low
        with pytest.raises(
            ValueError, match="Confidence level must be between 0 and 1"
        ):
            PredictionIntervalRequest(
                training_scores=training_scores, confidence_level=-0.1
            )

        # Too high
        with pytest.raises(
            ValueError, match="Confidence level must be between 0 and 1"
        ):
            PredictionIntervalRequest(
                training_scores=training_scores, confidence_level=1.1
            )

    def test_boundary_confidence_levels(self):
        """Test boundary confidence levels."""
        training_scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        # Valid boundary values
        for confidence_level in [0.0, 0.5, 0.9, 0.99, 1.0]:
            request = PredictionIntervalRequest(
                training_scores=training_scores, confidence_level=confidence_level
            )
            assert request.confidence_level == confidence_level


class TestUncertaintyIntegration:
    """Test suite for uncertainty integration scenarios."""

    def test_complete_uncertainty_quantification_workflow(self):
        """Test complete uncertainty quantification workflow."""
        # Create detection results for uncertainty analysis
        detection_results = [
            DetectionResult(
                id=str(uuid4()),
                detector_id="uncertainty_detector",
                dataset_id="test_dataset",
                anomaly_score=AnomalyScore(0.7 + i * 0.05),
                is_anomaly=i % 2 == 0,
                sample_index=i,
            )
            for i in range(20)
        ]

        # Create uncertainty request
        uncertainty_request = UncertaintyRequest(
            detection_results=detection_results,
            method="bootstrap",
            confidence_level=0.95,
            include_prediction_intervals=True,
            include_entropy=True,
            n_bootstrap=1000,
        )

        # Create uncertainty response
        confidence_intervals = {
            "mean_score": ConfidenceInterval(
                lower=0.72, upper=0.88, confidence_level=0.95
            ),
            "std_score": ConfidenceInterval(
                lower=0.25, upper=0.35, confidence_level=0.95
            ),
            "median_score": ConfidenceInterval(
                lower=0.75, upper=0.85, confidence_level=0.95
            ),
        }

        uncertainty_metrics = {
            "mean_score": 0.8,
            "std_score": 0.3,
            "median_score": 0.8,
            "coefficient_of_variation": 0.375,
            "quantile_95": 0.95,
            "quantile_05": 0.65,
        }

        additional_metrics = {
            "entropy": 0.55,
            "variance": 0.09,
            "skewness": 0.1,
            "kurtosis": -0.2,
            "prediction_interval_width": 0.16,
        }

        uncertainty_response = UncertaintyResponse(
            confidence_intervals=confidence_intervals,
            uncertainty_metrics=uncertainty_metrics,
            additional_metrics=additional_metrics,
            method="bootstrap",
            confidence_level=0.95,
            n_samples=20,
        )

        # Verify workflow consistency
        assert (
            len(uncertainty_request.detection_results) == uncertainty_response.n_samples
        )
        assert uncertainty_request.method == uncertainty_response.method
        assert (
            uncertainty_request.confidence_level
            == uncertainty_response.confidence_level
        )
        assert uncertainty_request.include_entropy is True
        assert "entropy" in uncertainty_response.additional_metrics
        assert uncertainty_request.include_prediction_intervals is True
        assert "prediction_interval_width" in uncertainty_response.additional_metrics

        # Test response methods
        summary = uncertainty_response.get_summary()
        assert summary["method"] == "bootstrap"
        assert summary["n_samples"] == 20
        assert summary["mean_score"] == 0.8
        assert summary["entropy"] == 0.55
        assert "mean_score_ci_width" in summary

        response_dict = uncertainty_response.to_dict()
        assert "confidence_intervals" in response_dict
        assert "uncertainty_metrics" in response_dict
        assert "additional_metrics" in response_dict

    def test_ensemble_uncertainty_workflow(self):
        """Test ensemble uncertainty quantification workflow."""
        # Create ensemble results (4 models, 15 samples each)
        ensemble_results = []
        for model_idx in range(4):
            model_results = []
            for sample_idx in range(15):
                # Add some variation between models
                base_score = 0.5 + sample_idx * 0.03
                model_variation = (model_idx - 1.5) * 0.1  # Center around 0
                final_score = max(0.1, min(0.9, base_score + model_variation))

                result = DetectionResult(
                    id=str(uuid4()),
                    detector_id=f"ensemble_detector_{model_idx}",
                    dataset_id="ensemble_dataset",
                    anomaly_score=AnomalyScore(final_score),
                    is_anomaly=final_score > 0.65,
                    sample_index=sample_idx,
                )
                model_results.append(result)
            ensemble_results.append(model_results)

        # Create ensemble uncertainty request
        ensemble_request = EnsembleUncertaintyRequest(
            ensemble_results=ensemble_results,
            method="bootstrap",
            confidence_level=0.95,
            include_disagreement=True,
        )

        # Create ensemble uncertainty response
        ensemble_metrics = {
            "ensemble_mean": 0.72,
            "ensemble_std": 0.18,
            "ensemble_median": 0.71,
            "aleatoric_uncertainty": 0.12,
            "epistemic_uncertainty": 0.06,
            "total_uncertainty": 0.18,
        }

        model_uncertainties = [
            {
                "model_id": i,
                "metrics": {
                    "mean": 0.70 + i * 0.02,
                    "std": 0.15 + i * 0.01,
                    "median": 0.69 + i * 0.02,
                },
            }
            for i in range(4)
        ]

        disagreement_metrics = {
            "variance": 0.032,
            "range": 0.25,
            "disagreement_rate": 0.33,
            "pairwise_correlation": 0.85,
            "consensus_score": 0.67,
        }

        ensemble_response = EnsembleUncertaintyResponse(
            ensemble_metrics=ensemble_metrics,
            model_uncertainties=model_uncertainties,
            disagreement_metrics=disagreement_metrics,
            method="bootstrap",
            confidence_level=0.95,
            n_models=4,
        )

        # Verify workflow consistency
        assert len(ensemble_request.ensemble_results) == ensemble_response.n_models
        assert ensemble_request.method == ensemble_response.method
        assert ensemble_request.confidence_level == ensemble_response.confidence_level
        assert ensemble_request.include_disagreement is True
        assert len(ensemble_response.disagreement_metrics) > 0
        assert len(ensemble_response.model_uncertainties) == 4

        # Test response methods
        summary = ensemble_response.get_summary()
        assert summary["method"] == "bootstrap"
        assert summary["n_models"] == 4
        assert summary["ensemble_mean"] == 0.72
        assert summary["aleatoric_uncertainty"] == 0.12
        assert summary["disagreement_rate"] == 0.33

        response_dict = ensemble_response.to_dict()
        assert "ensemble_metrics" in response_dict
        assert "model_uncertainties" in response_dict
        assert "disagreement_metrics" in response_dict

    def test_bootstrap_uncertainty_workflow(self):
        """Test bootstrap uncertainty quantification workflow."""
        # Create bootstrap request with varying scores
        scores = [
            0.1,
            0.15,
            0.2,
            0.25,
            0.3,
            0.35,
            0.4,
            0.45,
            0.5,
            0.55,
            0.6,
            0.65,
            0.7,
            0.75,
            0.8,
            0.85,
            0.9,
            0.45,
            0.65,
            0.75,
        ]

        bootstrap_request = BootstrapRequest(
            scores=scores, confidence_level=0.95, n_bootstrap=2000, statistic="mean"
        )

        # Test different statistics
        statistics = ["mean", "median", "std", "var"]
        for statistic in statistics:
            stat_request = BootstrapRequest(
                scores=scores,
                confidence_level=0.95,
                n_bootstrap=1000,
                statistic=statistic,
            )
            assert stat_request.statistic == statistic
            assert stat_request.n_bootstrap == 1000

        # Verify request properties
        assert bootstrap_request.scores == scores
        assert bootstrap_request.confidence_level == 0.95
        assert bootstrap_request.n_bootstrap == 2000
        assert bootstrap_request.statistic == "mean"

    def test_bayesian_uncertainty_workflow(self):
        """Test Bayesian uncertainty quantification workflow."""
        # Create Bayesian request with binary anomaly indicators
        binary_scores = [0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1]

        bayesian_request = BayesianRequest(
            binary_scores=binary_scores,
            confidence_level=0.95,
            prior_alpha=1.0,
            prior_beta=1.0,
        )

        # Test different priors
        priors = [
            (0.5, 0.5),  # Jeffreys prior
            (1.0, 1.0),  # Uniform prior
            (2.0, 2.0),  # Symmetric informative prior
            (3.0, 7.0),  # Asymmetric informative prior
        ]

        for alpha, beta in priors:
            prior_request = BayesianRequest(
                binary_scores=binary_scores,
                confidence_level=0.95,
                prior_alpha=alpha,
                prior_beta=beta,
            )
            assert prior_request.prior_alpha == alpha
            assert prior_request.prior_beta == beta

        # Verify request properties
        assert bayesian_request.binary_scores == binary_scores
        assert bayesian_request.confidence_level == 0.95
        assert bayesian_request.prior_alpha == 1.0
        assert bayesian_request.prior_beta == 1.0

    def test_prediction_interval_workflow(self):
        """Test prediction interval workflow."""
        # Create training scores for calibration
        training_scores = [
            0.05,
            0.1,
            0.15,
            0.2,
            0.25,
            0.3,
            0.35,
            0.4,
            0.45,
            0.5,
            0.55,
            0.6,
            0.65,
            0.7,
            0.75,
            0.8,
            0.85,
            0.9,
            0.95,
            0.42,
        ]

        prediction_request = PredictionIntervalRequest(
            training_scores=training_scores, confidence_level=0.95
        )

        # Test different confidence levels
        confidence_levels = [0.5, 0.8, 0.9, 0.95, 0.99]
        for confidence_level in confidence_levels:
            conf_request = PredictionIntervalRequest(
                training_scores=training_scores, confidence_level=confidence_level
            )
            assert conf_request.confidence_level == confidence_level

        # Verify request properties
        assert prediction_request.training_scores == training_scores
        assert prediction_request.confidence_level == 0.95
        assert len(prediction_request.training_scores) >= 10

    def test_dataclass_field_access(self):
        """Test dataclass field access and modification."""
        # Test UncertaintyRequest
        detection_results = [
            DetectionResult(
                id=str(uuid4()),
                detector_id="test_detector",
                dataset_id="test_dataset",
                anomaly_score=AnomalyScore(0.7),
                is_anomaly=False,
                sample_index=0,
            )
        ]

        uncertainty_request = UncertaintyRequest(detection_results=detection_results)

        # Test field access
        assert hasattr(uncertainty_request, "detection_results")
        assert hasattr(uncertainty_request, "method")
        assert hasattr(uncertainty_request, "confidence_level")
        assert hasattr(uncertainty_request, "include_prediction_intervals")
        assert hasattr(uncertainty_request, "include_entropy")
        assert hasattr(uncertainty_request, "n_bootstrap")

        # Test field modification
        uncertainty_request.method = "normal"
        assert uncertainty_request.method == "normal"

        uncertainty_request.confidence_level = 0.99
        assert uncertainty_request.confidence_level == 0.99

        uncertainty_request.n_bootstrap = 500
        assert uncertainty_request.n_bootstrap == 500

    def test_edge_cases_and_boundary_conditions(self):
        """Test edge cases and boundary conditions."""
        # Minimum detection results
        min_detection_results = [
            DetectionResult(
                id=str(uuid4()),
                detector_id="min_detector",
                dataset_id="min_dataset",
                anomaly_score=AnomalyScore(0.5),
                is_anomaly=False,
                sample_index=0,
            )
        ]

        min_request = UncertaintyRequest(
            detection_results=min_detection_results,
            confidence_level=0.0,  # Minimum confidence level
            n_bootstrap=100,  # Minimum n_bootstrap
        )
        assert len(min_request.detection_results) == 1
        assert min_request.confidence_level == 0.0
        assert min_request.n_bootstrap == 100

        # Maximum confidence level
        max_request = UncertaintyRequest(
            detection_results=min_detection_results,
            confidence_level=1.0,  # Maximum confidence level
        )
        assert max_request.confidence_level == 1.0

        # Minimum ensemble models
        min_ensemble = [[min_detection_results[0]], [min_detection_results[0]]]

        min_ensemble_request = EnsembleUncertaintyRequest(ensemble_results=min_ensemble)
        assert len(min_ensemble_request.ensemble_results) == 2

        # Minimum bootstrap scores
        min_bootstrap_request = BootstrapRequest(
            scores=[0.5],  # Single score
            n_bootstrap=100,  # Minimum n_bootstrap
        )
        assert len(min_bootstrap_request.scores) == 1
        assert min_bootstrap_request.n_bootstrap == 100

        # Minimum binary scores for Bayesian
        min_bayesian_request = BayesianRequest(
            binary_scores=[0],  # Single binary score
            prior_alpha=0.1,  # Small positive prior
            prior_beta=0.1,  # Small positive prior
        )
        assert len(min_bayesian_request.binary_scores) == 1
        assert min_bayesian_request.prior_alpha == 0.1
        assert min_bayesian_request.prior_beta == 0.1

        # Minimum training scores for prediction intervals
        min_training_scores = [
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
        ]  # Exactly 10 scores
        min_prediction_request = PredictionIntervalRequest(
            training_scores=min_training_scores
        )
        assert len(min_prediction_request.training_scores) == 10
