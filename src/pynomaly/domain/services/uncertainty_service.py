"""
Uncertainty quantification service for anomaly detection.

This module provides statistical methods for quantifying uncertainty in
anomaly detection predictions through confidence intervals and probabilistic
measures.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from scipy import stats

from pynomaly.domain.entities.detection_result import DetectionResult
from pynomaly.domain.value_objects.anomaly_score import AnomalyScore
from pynomaly.domain.value_objects.confidence_interval import ConfidenceInterval


class UncertaintyQuantificationProtocol(Protocol):
    """Protocol for uncertainty quantification methods."""

    def calculate_confidence_interval(
        self, predictions: np.ndarray, confidence_level: float = 0.95
    ) -> ConfidenceInterval:
        """Calculate confidence interval for predictions."""
        ...

    def calculate_prediction_uncertainty(
        self, scores: list[AnomalyScore], method: str = "bootstrap"
    ) -> dict[str, float]:
        """Calculate uncertainty metrics for predictions."""
        ...


class UncertaintyQuantificationService:
    """
    Domain service for uncertainty quantification in anomaly detection.

    Provides various statistical methods for estimating confidence intervals
    and uncertainty measures for anomaly detection predictions.
    """

    def __init__(self, random_seed: int | None = None):
        """
        Initialize uncertainty quantification service.

        Args:
            random_seed: Random seed for reproducible results
        """
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)

    def calculate_bootstrap_confidence_interval(
        self,
        scores: np.ndarray | list[float],
        confidence_level: float = 0.95,
        n_bootstrap: int = 1000,
        statistic_function: str = "mean",
    ) -> ConfidenceInterval:
        """
        Calculate confidence interval using bootstrap resampling.

        Args:
            scores: Array of anomaly scores
            confidence_level: Desired confidence level (0.0 to 1.0)
            n_bootstrap: Number of bootstrap samples
            statistic_function: Statistical function to apply ("mean", "median", "std")

        Returns:
            ConfidenceInterval based on bootstrap distribution
        """
        if isinstance(scores, list):
            scores = np.array(scores)

        if len(scores) == 0:
            raise ValueError("Cannot calculate confidence interval from empty scores")

        # Define statistic function
        stat_func_map = {
            "mean": np.mean,
            "median": np.median,
            "std": np.std,
            "var": np.var,
        }

        if statistic_function not in stat_func_map:
            raise ValueError(f"Unknown statistic function: {statistic_function}")

        stat_func = stat_func_map[statistic_function]

        # Bootstrap resampling
        bootstrap_stats = []
        for _ in range(n_bootstrap):
            # Sample with replacement
            bootstrap_sample = np.random.choice(scores, size=len(scores), replace=True)
            bootstrap_stats.append(stat_func(bootstrap_sample))

        bootstrap_stats = np.array(bootstrap_stats)

        return ConfidenceInterval.from_samples(
            samples=bootstrap_stats,
            confidence_level=confidence_level,
            method=f"bootstrap_{statistic_function}",
        )

    def calculate_bayesian_confidence_interval(
        self,
        scores: np.ndarray | list[float],
        confidence_level: float = 0.95,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
    ) -> ConfidenceInterval:
        """
        Calculate Bayesian confidence interval assuming Beta prior.

        Args:
            scores: Array of binary anomaly labels (0 or 1)
            confidence_level: Desired confidence level
            prior_alpha: Alpha parameter of Beta prior
            prior_beta: Beta parameter of Beta prior

        Returns:
            ConfidenceInterval for anomaly rate
        """
        if isinstance(scores, list):
            scores = np.array(scores)

        # Convert to binary if needed (assuming threshold of 0.5)
        binary_scores = (scores >= 0.5).astype(int)

        n_anomalies = np.sum(binary_scores)
        n_total = len(binary_scores)

        # Update Beta distribution parameters
        posterior_alpha = prior_alpha + n_anomalies
        posterior_beta = prior_beta + n_total - n_anomalies

        # Calculate confidence interval using Beta distribution
        alpha = 1 - confidence_level
        lower = stats.beta.ppf(alpha / 2, posterior_alpha, posterior_beta)
        upper = stats.beta.ppf(1 - alpha / 2, posterior_alpha, posterior_beta)

        return ConfidenceInterval.from_bounds(
            lower=lower,
            upper=upper,
            confidence_level=confidence_level,
            method="bayesian_beta",
        )

    def calculate_normal_confidence_interval(
        self, scores: np.ndarray | list[float], confidence_level: float = 0.95
    ) -> ConfidenceInterval:
        """
        Calculate confidence interval assuming normal distribution.

        Args:
            scores: Array of anomaly scores
            confidence_level: Desired confidence level

        Returns:
            ConfidenceInterval based on normal distribution
        """
        if isinstance(scores, list):
            scores = np.array(scores)

        if len(scores) == 0:
            raise ValueError("Cannot calculate confidence interval from empty scores")

        mean = np.mean(scores)
        std = np.std(scores, ddof=1)  # Sample standard deviation
        n = len(scores)

        # Standard error of the mean
        sem = std / np.sqrt(n)

        # t-distribution critical value
        alpha = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha / 2, df=n - 1)

        # Margin of error
        margin_of_error = t_critical * sem

        return ConfidenceInterval.from_center_and_margin(
            center=mean,
            margin_of_error=margin_of_error,
            confidence_level=confidence_level,
            method="normal_t_distribution",
        )

    def calculate_prediction_uncertainty(
        self, detection_results: list[DetectionResult], method: str = "bootstrap"
    ) -> dict[str, float | ConfidenceInterval]:
        """
        Calculate comprehensive uncertainty metrics for detection results.

        Args:
            detection_results: List of detection results
            method: Uncertainty calculation method

        Returns:
            Dictionary containing uncertainty metrics
        """
        if not detection_results:
            raise ValueError("Cannot calculate uncertainty from empty results")

        # Extract scores
        scores = np.array([result.score.value for result in detection_results])

        # Calculate various uncertainty metrics
        uncertainty_metrics = {}

        # Basic statistics
        uncertainty_metrics["mean_score"] = float(np.mean(scores))
        uncertainty_metrics["std_score"] = float(np.std(scores))
        uncertainty_metrics["variance_score"] = float(np.var(scores))
        uncertainty_metrics["coefficient_of_variation"] = float(
            np.std(scores) / np.mean(scores)
        )

        # Confidence intervals
        if method == "bootstrap":
            ci_mean = self.calculate_bootstrap_confidence_interval(
                scores, statistic_function="mean"
            )
            ci_std = self.calculate_bootstrap_confidence_interval(
                scores, statistic_function="std"
            )
        elif method == "normal":
            ci_mean = self.calculate_normal_confidence_interval(scores)
            ci_std = None
        elif method == "bayesian":
            ci_mean = self.calculate_bayesian_confidence_interval(scores)
            ci_std = None
        else:
            raise ValueError(f"Unknown uncertainty method: {method}")

        uncertainty_metrics["confidence_interval_mean"] = ci_mean
        if ci_std is not None:
            uncertainty_metrics["confidence_interval_std"] = ci_std

        # Prediction interval (for individual predictions)
        uncertainty_metrics[
            "prediction_interval"
        ] = self._calculate_prediction_interval(scores)

        # Entropy-based uncertainty (if applicable)
        uncertainty_metrics["entropy"] = self._calculate_entropy(scores)

        return uncertainty_metrics

    def calculate_ensemble_uncertainty(
        self, ensemble_scores: list[list[float]], confidence_level: float = 0.95
    ) -> dict[str, float | ConfidenceInterval]:
        """
        Calculate uncertainty for ensemble predictions.

        Args:
            ensemble_scores: List of score arrays from different models
            confidence_level: Desired confidence level

        Returns:
            Dictionary containing ensemble uncertainty metrics
        """
        if not ensemble_scores:
            raise ValueError("Cannot calculate ensemble uncertainty from empty scores")

        # Convert to numpy array
        ensemble_array = np.array(ensemble_scores)

        # Calculate statistics across ensemble
        mean_scores = np.mean(ensemble_array, axis=0)
        std_scores = np.std(ensemble_array, axis=0)

        # Overall uncertainty metrics
        uncertainty_metrics = {
            "ensemble_mean": float(np.mean(mean_scores)),
            "ensemble_std": float(np.mean(std_scores)),
            "total_variance": float(np.var(ensemble_array)),
            "aleatoric_uncertainty": float(
                np.mean(std_scores**2)
            ),  # Data uncertainty
            "epistemic_uncertainty": float(np.var(mean_scores)),  # Model uncertainty
        }

        # Confidence interval for ensemble mean
        ci_ensemble = self.calculate_bootstrap_confidence_interval(
            mean_scores, confidence_level=confidence_level
        )
        uncertainty_metrics["ensemble_confidence_interval"] = ci_ensemble

        return uncertainty_metrics

    def _calculate_prediction_interval(
        self, scores: np.ndarray, confidence_level: float = 0.95
    ) -> ConfidenceInterval:
        """Calculate prediction interval for individual predictions."""
        mean = np.mean(scores)
        std = np.std(scores, ddof=1)
        n = len(scores)

        # Prediction interval is wider than confidence interval
        # because it accounts for both sampling uncertainty and individual variation
        alpha = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha / 2, df=n - 1)

        # Prediction interval margin
        margin = t_critical * std * np.sqrt(1 + 1 / n)

        return ConfidenceInterval.from_center_and_margin(
            center=mean,
            margin_of_error=margin,
            confidence_level=confidence_level,
            method="prediction_interval",
        )

    def _calculate_entropy(self, scores: np.ndarray) -> float:
        """Calculate entropy-based uncertainty measure."""
        # Normalize scores to [0, 1] if not already
        normalized_scores = (scores - np.min(scores)) / (
            np.max(scores) - np.min(scores)
        )

        # Avoid log(0) by adding small epsilon
        epsilon = 1e-15
        normalized_scores = np.clip(normalized_scores, epsilon, 1 - epsilon)

        # Calculate entropy
        entropy = -np.mean(
            normalized_scores * np.log2(normalized_scores)
            + (1 - normalized_scores) * np.log2(1 - normalized_scores)
        )

        return float(entropy)

    def calculate_credible_interval(
        self, posterior_samples: np.ndarray, confidence_level: float = 0.95
    ) -> ConfidenceInterval:
        """
        Calculate Bayesian credible interval from posterior samples.

        Args:
            posterior_samples: Samples from posterior distribution
            confidence_level: Desired confidence level

        Returns:
            ConfidenceInterval representing credible interval
        """
        return ConfidenceInterval.from_samples(
            samples=posterior_samples,
            confidence_level=confidence_level,
            method="bayesian_credible_interval",
        )

    def calculate_highest_density_interval(
        self, samples: np.ndarray, confidence_level: float = 0.95
    ) -> ConfidenceInterval:
        """
        Calculate highest density interval (HDI).

        The HDI is the shortest interval containing the specified probability mass.

        Args:
            samples: Array of samples from distribution
            confidence_level: Desired confidence level

        Returns:
            ConfidenceInterval representing HDI
        """
        sorted_samples = np.sort(samples)
        n = len(sorted_samples)

        # Number of samples to include in interval
        n_include = int(np.ceil(confidence_level * n))

        # Find the shortest interval containing n_include samples
        interval_widths = (
            sorted_samples[n_include - 1 :] - sorted_samples[: n - n_include + 1]
        )
        min_width_idx = np.argmin(interval_widths)

        lower = sorted_samples[min_width_idx]
        upper = sorted_samples[min_width_idx + n_include - 1]

        return ConfidenceInterval.from_bounds(
            lower=lower,
            upper=upper,
            confidence_level=confidence_level,
            method="highest_density_interval",
        )
