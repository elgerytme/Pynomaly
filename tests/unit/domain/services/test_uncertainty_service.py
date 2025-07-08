"""
Unit tests for uncertainty quantification service.

Tests the UncertaintyQuantificationService domain service functionality
including bootstrap, Bayesian, and normal confidence intervals.
"""


import numpy as np
import pytest
from pynomaly.domain.entities.detection_result import DetectionResult
from pynomaly.domain.services.uncertainty_service import (
    UncertaintyQuantificationService,
)
from pynomaly.domain.value_objects.anomaly_score import AnomalyScore
from pynomaly.domain.value_objects.confidence_interval import ConfidenceInterval


class TestUncertaintyQuantificationService:
    """Test cases for UncertaintyQuantificationService."""

    @pytest.fixture
    def uncertainty_service(self):
        """Create uncertainty service with fixed random seed."""
        return UncertaintyQuantificationService(random_seed=42)

    @pytest.fixture
    def sample_scores(self):
        """Create sample anomaly scores for testing."""
        return np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

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

    def test_bootstrap_confidence_interval_mean(
        self, uncertainty_service, sample_scores
    ):
        """Test bootstrap confidence interval for mean."""
        ci = uncertainty_service.calculate_bootstrap_confidence_interval(
            scores=sample_scores,
            confidence_level=0.95,
            n_bootstrap=100,
            statistic_function="mean",
        )

        assert isinstance(ci, ConfidenceInterval)
        assert ci.confidence_level == 0.95
        assert "bootstrap_mean" in ci.method
        assert ci.lower <= ci.upper

        # The true mean is 0.55, CI should contain it
        true_mean = np.mean(sample_scores)
        assert ci.contains(true_mean)

    def test_bootstrap_confidence_interval_median(
        self, uncertainty_service, sample_scores
    ):
        """Test bootstrap confidence interval for median."""
        ci = uncertainty_service.calculate_bootstrap_confidence_interval(
            scores=sample_scores,
            confidence_level=0.90,
            n_bootstrap=100,
            statistic_function="median",
        )

        assert isinstance(ci, ConfidenceInterval)
        assert ci.confidence_level == 0.90
        assert "bootstrap_median" in ci.method

        # The true median should be contained in the CI
        true_median = np.median(sample_scores)
        assert ci.contains(true_median)

    def test_bootstrap_confidence_interval_std(
        self, uncertainty_service, sample_scores
    ):
        """Test bootstrap confidence interval for standard deviation."""
        ci = uncertainty_service.calculate_bootstrap_confidence_interval(
            scores=sample_scores,
            confidence_level=0.95,
            n_bootstrap=100,
            statistic_function="std",
        )

        assert isinstance(ci, ConfidenceInterval)
        assert "bootstrap_std" in ci.method
        assert ci.lower >= 0  # Standard deviation is non-negative

    def test_bootstrap_invalid_statistic(self, uncertainty_service, sample_scores):
        """Test bootstrap with invalid statistic function."""
        with pytest.raises(ValueError, match="Unknown statistic function"):
            uncertainty_service.calculate_bootstrap_confidence_interval(
                scores=sample_scores, statistic_function="invalid_stat"
            )

    def test_bootstrap_empty_scores(self, uncertainty_service):
        """Test bootstrap with empty scores."""
        with pytest.raises(
            ValueError, match="Cannot calculate confidence interval from empty scores"
        ):
            uncertainty_service.calculate_bootstrap_confidence_interval(scores=[])

    def test_bayesian_confidence_interval(self, uncertainty_service):
        """Test Bayesian confidence interval calculation."""
        # Binary scores: 3 anomalies out of 10 samples
        binary_scores = [0, 0, 1, 0, 1, 0, 0, 1, 0, 0]

        ci = uncertainty_service.calculate_bayesian_confidence_interval(
            scores=binary_scores, confidence_level=0.95, prior_alpha=1.0, prior_beta=1.0
        )

        assert isinstance(ci, ConfidenceInterval)
        assert ci.confidence_level == 0.95
        assert "bayesian_beta" in ci.method
        assert 0.0 <= ci.lower <= ci.upper <= 1.0

        # With uniform prior and 3/10 anomalies, posterior should be around 0.3
        assert ci.contains(0.3)

    def test_bayesian_with_continuous_scores(self, uncertainty_service, sample_scores):
        """Test Bayesian interval with continuous scores (converted to binary)."""
        ci = uncertainty_service.calculate_bayesian_confidence_interval(
            scores=sample_scores,  # Will be converted to binary with threshold 0.5
            confidence_level=0.95,
        )

        assert isinstance(ci, ConfidenceInterval)
        assert 0.0 <= ci.lower <= ci.upper <= 1.0

    def test_normal_confidence_interval(self, uncertainty_service, sample_scores):
        """Test normal distribution confidence interval."""
        ci = uncertainty_service.calculate_normal_confidence_interval(
            scores=sample_scores, confidence_level=0.95
        )

        assert isinstance(ci, ConfidenceInterval)
        assert ci.confidence_level == 0.95
        assert "normal_t_distribution" in ci.method

        # Should contain the sample mean
        sample_mean = np.mean(sample_scores)
        assert ci.contains(sample_mean)

    def test_normal_confidence_interval_empty_scores(self, uncertainty_service):
        """Test normal CI with empty scores."""
        with pytest.raises(
            ValueError, match="Cannot calculate confidence interval from empty scores"
        ):
            uncertainty_service.calculate_normal_confidence_interval(scores=[])

    def test_prediction_uncertainty_bootstrap(
        self, uncertainty_service, sample_detection_results
    ):
        """Test prediction uncertainty calculation with bootstrap method."""
        uncertainty_metrics = uncertainty_service.calculate_prediction_uncertainty(
            detection_results=sample_detection_results, method="bootstrap"
        )

        assert isinstance(uncertainty_metrics, dict)

        # Check basic statistics
        assert "mean_score" in uncertainty_metrics
        assert "std_score" in uncertainty_metrics
        assert "variance_score" in uncertainty_metrics
        assert "coefficient_of_variation" in uncertainty_metrics

        # Check confidence intervals
        assert "confidence_interval_mean" in uncertainty_metrics
        assert "confidence_interval_std" in uncertainty_metrics
        assert isinstance(
            uncertainty_metrics["confidence_interval_mean"], ConfidenceInterval
        )

        # Check additional metrics
        assert "prediction_interval" in uncertainty_metrics
        assert "entropy" in uncertainty_metrics

    def test_prediction_uncertainty_normal(
        self, uncertainty_service, sample_detection_results
    ):
        """Test prediction uncertainty calculation with normal method."""
        uncertainty_metrics = uncertainty_service.calculate_prediction_uncertainty(
            detection_results=sample_detection_results, method="normal"
        )

        assert isinstance(uncertainty_metrics, dict)
        assert "confidence_interval_mean" in uncertainty_metrics
        assert (
            "confidence_interval_std" not in uncertainty_metrics
        )  # Not available for normal method

    def test_prediction_uncertainty_bayesian(
        self, uncertainty_service, sample_detection_results
    ):
        """Test prediction uncertainty calculation with Bayesian method."""
        uncertainty_metrics = uncertainty_service.calculate_prediction_uncertainty(
            detection_results=sample_detection_results, method="bayesian"
        )

        assert isinstance(uncertainty_metrics, dict)
        assert "confidence_interval_mean" in uncertainty_metrics

    def test_prediction_uncertainty_invalid_method(
        self, uncertainty_service, sample_detection_results
    ):
        """Test prediction uncertainty with invalid method."""
        with pytest.raises(ValueError, match="Unknown uncertainty method"):
            uncertainty_service.calculate_prediction_uncertainty(
                detection_results=sample_detection_results, method="invalid_method"
            )

    def test_prediction_uncertainty_empty_results(self, uncertainty_service):
        """Test prediction uncertainty with empty results."""
        with pytest.raises(
            ValueError, match="Cannot calculate uncertainty from empty results"
        ):
            uncertainty_service.calculate_prediction_uncertainty(detection_results=[])

    def test_ensemble_uncertainty(self, uncertainty_service):
        """Test ensemble uncertainty calculation."""
        # Create ensemble scores (3 models, 5 samples each)
        ensemble_scores = [
            [0.1, 0.2, 0.3, 0.4, 0.5],  # Model 1
            [0.15, 0.25, 0.35, 0.45, 0.55],  # Model 2
            [0.05, 0.15, 0.25, 0.35, 0.45],  # Model 3
        ]

        uncertainty_metrics = uncertainty_service.calculate_ensemble_uncertainty(
            ensemble_scores=ensemble_scores, confidence_level=0.95
        )

        assert isinstance(uncertainty_metrics, dict)

        # Check ensemble metrics
        assert "ensemble_mean" in uncertainty_metrics
        assert "ensemble_std" in uncertainty_metrics
        assert "total_variance" in uncertainty_metrics
        assert "aleatoric_uncertainty" in uncertainty_metrics  # Data uncertainty
        assert "epistemic_uncertainty" in uncertainty_metrics  # Model uncertainty
        assert "ensemble_confidence_interval" in uncertainty_metrics

        # Check that values are reasonable
        assert uncertainty_metrics["ensemble_mean"] > 0
        assert uncertainty_metrics["ensemble_std"] >= 0
        assert uncertainty_metrics["total_variance"] >= 0
        assert uncertainty_metrics["aleatoric_uncertainty"] >= 0
        assert uncertainty_metrics["epistemic_uncertainty"] >= 0

    def test_ensemble_uncertainty_empty_scores(self, uncertainty_service):
        """Test ensemble uncertainty with empty scores."""
        with pytest.raises(
            ValueError, match="Cannot calculate ensemble uncertainty from empty scores"
        ):
            uncertainty_service.calculate_ensemble_uncertainty(ensemble_scores=[])

    def test_credible_interval(self, uncertainty_service):
        """Test Bayesian credible interval calculation."""
        # Sample from a known distribution
        posterior_samples = np.random.normal(0.5, 0.1, 1000)

        ci = uncertainty_service.calculate_credible_interval(
            posterior_samples=posterior_samples, confidence_level=0.95
        )

        assert isinstance(ci, ConfidenceInterval)
        assert ci.confidence_level == 0.95
        assert "bayesian_credible_interval" in ci.method

        # Should approximately contain the true mean
        assert abs(ci.midpoint() - 0.5) < 0.1

    def test_highest_density_interval(self, uncertainty_service):
        """Test highest density interval calculation."""
        # Create samples from a known distribution
        samples = np.random.normal(0, 1, 1000)

        ci = uncertainty_service.calculate_highest_density_interval(
            samples=samples, confidence_level=0.95
        )

        assert isinstance(ci, ConfidenceInterval)
        assert ci.confidence_level == 0.95
        assert "highest_density_interval" in ci.method

        # HDI should be shorter than or equal to percentile-based interval
        percentile_ci = ConfidenceInterval.from_samples(
            samples=samples, confidence_level=0.95
        )

        # HDI width should be <= percentile interval width (for symmetric distributions)
        assert (
            ci.width() <= percentile_ci.width() + 1e-6
        )  # Small tolerance for numerical precision

    def test_prediction_interval_calculation(self, uncertainty_service, sample_scores):
        """Test prediction interval calculation."""
        ci = uncertainty_service._calculate_prediction_interval(
            scores=sample_scores, confidence_level=0.95
        )

        assert isinstance(ci, ConfidenceInterval)
        assert "prediction_interval" in ci.method

        # Prediction interval should be wider than confidence interval
        conf_ci = uncertainty_service.calculate_normal_confidence_interval(
            scores=sample_scores, confidence_level=0.95
        )

        assert ci.width() > conf_ci.width()

    def test_entropy_calculation(self, uncertainty_service):
        """Test entropy-based uncertainty calculation."""
        # Test with uniform distribution (high entropy)
        uniform_scores = np.full(100, 0.5)
        uniform_entropy = uncertainty_service._calculate_entropy(uniform_scores)

        # Test with extreme values (low entropy)
        extreme_scores = np.concatenate([np.zeros(50), np.ones(50)])
        extreme_entropy = uncertainty_service._calculate_entropy(extreme_scores)

        # Extreme values should have lower entropy than uniform
        assert extreme_entropy < uniform_entropy

        # Test with single value (should be 0 entropy)
        single_scores = np.full(100, 0.8)
        single_entropy = uncertainty_service._calculate_entropy(single_scores)
        assert single_entropy == 0.0

    def test_random_seed_reproducibility(self):
        """Test that random seed produces reproducible results."""
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        # Create two services with same seed
        service1 = UncertaintyQuantificationService(random_seed=42)
        service2 = UncertaintyQuantificationService(random_seed=42)

        ci1 = service1.calculate_bootstrap_confidence_interval(
            scores=scores, n_bootstrap=100
        )

        ci2 = service2.calculate_bootstrap_confidence_interval(
            scores=scores, n_bootstrap=100
        )

        # Results should be identical with same seed
        assert ci1.lower == ci2.lower
        assert ci1.upper == ci2.upper

    def test_list_input_conversion(self, uncertainty_service):
        """Test that list inputs are properly converted to numpy arrays."""
        scores_list = [0.1, 0.2, 0.3, 0.4, 0.5]
        scores_array = np.array(scores_list)

        ci_list = uncertainty_service.calculate_bootstrap_confidence_interval(
            scores=scores_list, n_bootstrap=100
        )

        ci_array = uncertainty_service.calculate_bootstrap_confidence_interval(
            scores=scores_array, n_bootstrap=100
        )

        # Results should be the same regardless of input type
        assert ci_list.lower == ci_array.lower
        assert ci_list.upper == ci_array.upper
