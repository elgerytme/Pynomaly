"""
Infrastructure adapter for uncertainty quantification integration.

This module provides adapters for integrating external uncertainty
quantification libraries and statistical methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy import stats
from sklearn.utils import resample

from pynomaly.domain.value_objects.confidence_interval import ConfidenceInterval


class UncertaintyAdapterProtocol(ABC):
    """Protocol for uncertainty quantification adapters."""

    @abstractmethod
    def calculate_bootstrap_interval(
        self,
        data: np.ndarray,
        confidence_level: float,
        n_bootstrap: int,
        statistic_func: str,
    ) -> ConfidenceInterval:
        """Calculate bootstrap confidence interval."""
        ...

    @abstractmethod
    def calculate_bayesian_interval(
        self,
        successes: int,
        trials: int,
        confidence_level: float,
        prior_alpha: float,
        prior_beta: float,
    ) -> ConfidenceInterval:
        """Calculate Bayesian confidence interval."""
        ...

    @abstractmethod
    def calculate_normal_interval(
        self, data: np.ndarray, confidence_level: float
    ) -> ConfidenceInterval:
        """Calculate normal-based confidence interval."""
        ...


class ScipyUncertaintyAdapter(UncertaintyAdapterProtocol):
    """
    Uncertainty quantification adapter using SciPy statistical functions.

    Provides statistical uncertainty quantification methods using SciPy's
    robust statistical functions and distributions.
    """

    def __init__(self, random_seed: int | None = None):
        """
        Initialize SciPy uncertainty adapter.

        Args:
            random_seed: Random seed for reproducible results
        """
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)

    def calculate_bootstrap_interval(
        self,
        data: np.ndarray,
        confidence_level: float = 0.95,
        n_bootstrap: int = 1000,
        statistic_func: str = "mean",
    ) -> ConfidenceInterval:
        """
        Calculate bootstrap confidence interval using SciPy.

        Args:
            data: Input data array
            confidence_level: Desired confidence level
            n_bootstrap: Number of bootstrap samples
            statistic_func: Statistic function to apply

        Returns:
            ConfidenceInterval based on bootstrap distribution
        """
        if len(data) == 0:
            raise ValueError("Cannot calculate bootstrap interval from empty data")

        # Define statistic functions
        stat_funcs = {
            "mean": np.mean,
            "median": np.median,
            "std": lambda x: np.std(x, ddof=1),
            "var": lambda x: np.var(x, ddof=1),
            "min": np.min,
            "max": np.max,
            "percentile_25": lambda x: np.percentile(x, 25),
            "percentile_75": lambda x: np.percentile(x, 75),
        }

        if statistic_func not in stat_funcs:
            raise ValueError(f"Unknown statistic function: {statistic_func}")

        stat_func = stat_funcs[statistic_func]

        # Perform bootstrap resampling
        bootstrap_stats = []
        for _ in range(n_bootstrap):
            # Resample with replacement
            bootstrap_sample = resample(
                data, n_samples=len(data), random_state=self.random_seed
            )
            bootstrap_stats.append(stat_func(bootstrap_sample))

        bootstrap_stats = np.array(bootstrap_stats)

        # Calculate percentile-based confidence interval
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        lower = np.percentile(bootstrap_stats, lower_percentile)
        upper = np.percentile(bootstrap_stats, upper_percentile)

        return ConfidenceInterval.from_bounds(
            lower=float(lower),
            upper=float(upper),
            confidence_level=confidence_level,
            method=f"bootstrap_{statistic_func}_scipy",
        )

    def calculate_bayesian_interval(
        self,
        successes: int,
        trials: int,
        confidence_level: float = 0.95,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
    ) -> ConfidenceInterval:
        """
        Calculate Bayesian confidence interval using Beta distribution.

        Args:
            successes: Number of successes (anomalies)
            trials: Total number of trials
            confidence_level: Desired confidence level
            prior_alpha: Alpha parameter of Beta prior
            prior_beta: Beta parameter of Beta prior

        Returns:
            ConfidenceInterval for success rate
        """
        if trials <= 0:
            raise ValueError("Number of trials must be positive")

        if successes < 0 or successes > trials:
            raise ValueError("Successes must be between 0 and trials")

        # Update Beta distribution parameters
        posterior_alpha = prior_alpha + successes
        posterior_beta = prior_beta + trials - successes

        # Calculate credible interval using Beta distribution
        alpha = 1 - confidence_level
        lower = stats.beta.ppf(alpha / 2, posterior_alpha, posterior_beta)
        upper = stats.beta.ppf(1 - alpha / 2, posterior_alpha, posterior_beta)

        return ConfidenceInterval.from_bounds(
            lower=float(lower),
            upper=float(upper),
            confidence_level=confidence_level,
            method="bayesian_beta_scipy",
        )

    def calculate_normal_interval(
        self, data: np.ndarray, confidence_level: float = 0.95
    ) -> ConfidenceInterval:
        """
        Calculate confidence interval assuming normal distribution.

        Args:
            data: Input data array
            confidence_level: Desired confidence level

        Returns:
            ConfidenceInterval based on normal distribution
        """
        if len(data) == 0:
            raise ValueError("Cannot calculate normal interval from empty data")

        # Calculate sample statistics
        mean = np.mean(data)
        std = np.std(data, ddof=1)  # Sample standard deviation
        n = len(data)

        # Standard error of the mean
        sem = std / np.sqrt(n)

        # t-distribution critical value (more appropriate for small samples)
        alpha = 1 - confidence_level
        if n > 30:
            # Use normal distribution for large samples
            critical_value = stats.norm.ppf(1 - alpha / 2)
        else:
            # Use t-distribution for small samples
            critical_value = stats.t.ppf(1 - alpha / 2, df=n - 1)

        # Calculate margin of error
        margin_of_error = critical_value * sem

        return ConfidenceInterval.from_center_and_margin(
            center=float(mean),
            margin_of_error=float(margin_of_error),
            confidence_level=confidence_level,
            method="normal_t_scipy" if n <= 30 else "normal_z_scipy",
        )

    def calculate_poisson_interval(
        self, count: int, exposure: float, confidence_level: float = 0.95
    ) -> ConfidenceInterval:
        """
        Calculate confidence interval for Poisson rate parameter.

        Args:
            count: Observed count
            exposure: Exposure time/amount
            confidence_level: Desired confidence level

        Returns:
            ConfidenceInterval for rate parameter
        """
        if exposure <= 0:
            raise ValueError("Exposure must be positive")

        if count < 0:
            raise ValueError("Count must be non-negative")

        alpha = 1 - confidence_level

        # Use exact Poisson confidence interval
        if count == 0:
            lower = 0.0
            upper = -np.log(alpha / 2) / exposure
        else:
            # Use chi-square distribution for exact intervals
            lower = stats.chi2.ppf(alpha / 2, 2 * count) / (2 * exposure)
            upper = stats.chi2.ppf(1 - alpha / 2, 2 * (count + 1)) / (2 * exposure)

        return ConfidenceInterval.from_bounds(
            lower=float(lower),
            upper=float(upper),
            confidence_level=confidence_level,
            method="poisson_exact_scipy",
        )

    def calculate_wilson_score_interval(
        self, successes: int, trials: int, confidence_level: float = 0.95
    ) -> ConfidenceInterval:
        """
        Calculate Wilson score confidence interval for proportions.

        More robust than normal approximation for small samples or extreme proportions.

        Args:
            successes: Number of successes
            trials: Total number of trials
            confidence_level: Desired confidence level

        Returns:
            ConfidenceInterval for proportion
        """
        if trials <= 0:
            raise ValueError("Number of trials must be positive")

        if successes < 0 or successes > trials:
            raise ValueError("Successes must be between 0 and trials")

        # Sample proportion
        p = successes / trials

        # Critical value from standard normal distribution
        alpha = 1 - confidence_level
        z = stats.norm.ppf(1 - alpha / 2)

        # Wilson score interval calculation
        denominator = 1 + z**2 / trials
        center = (p + z**2 / (2 * trials)) / denominator
        margin = (
            z * np.sqrt((p * (1 - p) + z**2 / (4 * trials)) / trials) / denominator
        )

        lower = max(0.0, center - margin)
        upper = min(1.0, center + margin)

        return ConfidenceInterval.from_bounds(
            lower=float(lower),
            upper=float(upper),
            confidence_level=confidence_level,
            method="wilson_score_scipy",
        )

    def calculate_prediction_interval(
        self,
        data: np.ndarray,
        confidence_level: float = 0.95,
        future_observations: int = 1,
    ) -> ConfidenceInterval:
        """
        Calculate prediction interval for future observations.

        Args:
            data: Historical data
            confidence_level: Desired confidence level
            future_observations: Number of future observations

        Returns:
            ConfidenceInterval for future observations
        """
        if len(data) == 0:
            raise ValueError("Cannot calculate prediction interval from empty data")

        # Sample statistics
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        n = len(data)

        # Prediction interval accounts for both sampling uncertainty
        # and natural variation in future observations
        alpha = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha / 2, df=n - 1)

        # Standard error for prediction
        prediction_se = std * np.sqrt(1 + future_observations / n)

        # Margin of error
        margin_of_error = t_critical * prediction_se

        return ConfidenceInterval.from_center_and_margin(
            center=float(mean),
            margin_of_error=float(margin_of_error),
            confidence_level=confidence_level,
            method="prediction_interval_scipy",
        )

    def calculate_tolerance_interval(
        self, data: np.ndarray, coverage: float = 0.95, confidence_level: float = 0.95
    ) -> ConfidenceInterval:
        """
        Calculate tolerance interval containing a specified proportion of the population.

        Args:
            data: Sample data
            coverage: Proportion of population to cover
            confidence_level: Confidence in the interval

        Returns:
            ConfidenceInterval representing tolerance interval
        """
        if len(data) == 0:
            raise ValueError("Cannot calculate tolerance interval from empty data")

        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)

        # Calculate tolerance factor (approximation for normal distribution)
        # This is a simplified calculation; exact calculation requires
        # non-central t-distribution
        alpha = 1 - confidence_level
        beta = 1 - coverage

        # Approximate tolerance factor
        z_coverage = stats.norm.ppf(1 - beta / 2)
        chi2_conf = stats.chi2.ppf(confidence_level, df=n - 1)

        # Tolerance factor approximation
        k = z_coverage * np.sqrt((n - 1) * (1 + 1 / n) / chi2_conf)

        # Tolerance interval
        lower = mean - k * std
        upper = mean + k * std

        return ConfidenceInterval.from_bounds(
            lower=float(lower),
            upper=float(upper),
            confidence_level=confidence_level,
            method=f"tolerance_interval_scipy_cov{coverage:.2f}",
        )
