"""Data Distribution value object for statistical distribution analysis."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import Field, validator

from packages.core.domain.abstractions.base_value_object import BaseValueObject


class DistributionType(str, Enum):
    """Types of statistical distributions."""
    
    NORMAL = "normal"
    UNIFORM = "uniform"
    EXPONENTIAL = "exponential"
    GAMMA = "gamma"
    BETA = "beta"
    BINOMIAL = "binomial"
    POISSON = "poisson"
    LOG_NORMAL = "log_normal"
    CHI_SQUARED = "chi_squared"
    T_DISTRIBUTION = "t_distribution"
    F_DISTRIBUTION = "f_distribution"
    WEIBULL = "weibull"
    PARETO = "pareto"
    LAPLACE = "laplace"
    CAUCHY = "cauchy"
    MULTIVARIATE_NORMAL = "multivariate_normal"
    UNKNOWN = "unknown"
    CUSTOM = "custom"


class DistributionTest(str, Enum):
    """Statistical tests for distribution fitting."""
    
    KOLMOGOROV_SMIRNOV = "kolmogorov_smirnov"
    ANDERSON_DARLING = "anderson_darling"
    SHAPIRO_WILK = "shapiro_wilk"
    JARQUE_BERA = "jarque_bera"
    DAGOSTINO = "dagostino"
    CHI_SQUARED_GOF = "chi_squared_gof"
    LILLIEFORS = "lilliefors"
    CRAMER_VON_MISES = "cramer_von_mises"


class DataDistribution(BaseValueObject):
    """Value object representing statistical distribution characteristics.
    
    This immutable value object encapsulates the statistical distribution
    properties of data, including fitted distributions, goodness-of-fit tests,
    and distribution parameters.
    
    Attributes:
        sample_size: Number of observations
        distribution_type: Best fitting distribution type
        parameters: Distribution parameters
        
        # Goodness of fit metrics
        ks_statistic: Kolmogorov-Smirnov test statistic
        ks_p_value: KS test p-value
        ad_statistic: Anderson-Darling test statistic
        ad_p_value: AD test p-value
        sw_statistic: Shapiro-Wilk test statistic
        sw_p_value: SW test p-value
        jb_statistic: Jarque-Bera test statistic
        jb_p_value: JB test p-value
        
        # Distribution parameters
        location: Location parameter (mu for normal)
        scale: Scale parameter (sigma for normal)
        shape: Shape parameter(s) for distributions
        degrees_freedom: Degrees of freedom for t, chi-squared, F distributions
        
        # Fitted distribution statistics
        log_likelihood: Log-likelihood of fitted distribution
        aic: Akaike Information Criterion
        bic: Bayesian Information Criterion
        confidence_level: Confidence level for parameter estimates
        parameter_confidence_intervals: CI for parameters
        
        # Distribution properties
        theoretical_mean: Theoretical mean of fitted distribution
        theoretical_variance: Theoretical variance
        theoretical_skewness: Theoretical skewness
        theoretical_kurtosis: Theoretical kurtosis
        
        # Empirical vs theoretical comparison
        mean_difference: Difference between empirical and theoretical mean
        variance_ratio: Ratio of empirical to theoretical variance
        qq_plot_correlation: Correlation coefficient from Q-Q plot
        pp_plot_correlation: Correlation coefficient from P-P plot
        
        # Distribution testing
        best_fit_test: Best test for this distribution type
        normality_p_value: Combined normality test p-value
        is_normal: Whether data appears normally distributed
        alternative_distributions: Other plausible distributions
        
        # Tail behavior
        left_tail_behavior: Behavior of left tail
        right_tail_behavior: Behavior of right tail
        tail_index: Heavy-tailedness measure
        extreme_value_index: Extreme value index
        
        # Multivariate properties (if applicable)
        correlation_matrix: Correlation matrix for multivariate data
        covariance_matrix: Covariance matrix
        mahalanobis_distances: Mahalanobis distances for outlier detection
        eigenvalues: Eigenvalues of covariance matrix
        condition_number: Condition number of covariance matrix
        
        # Data transformation recommendations
        transformation_needed: Whether transformation is recommended
        recommended_transformation: Suggested transformation type
        transformation_lambda: Box-Cox lambda parameter
        transformed_distribution: Distribution after transformation
        
        # Distribution metrics
        entropy: Differential entropy
        information_content: Information content
        complexity_measure: Distribution complexity measure
        goodness_of_fit_score: Overall goodness of fit (0-1)
        
        # Simulation and sampling
        random_seed: Seed for reproducible sampling
        sample_statistics: Statistics from generated samples
        bootstrap_confidence: Bootstrap confidence intervals
        monte_carlo_samples: Number of MC samples for estimation
    """
    
    sample_size: int = Field(..., gt=0)
    distribution_type: DistributionType
    parameters: dict[str, float] = Field(default_factory=dict)
    
    # Goodness of fit tests
    ks_statistic: Optional[float] = Field(None, ge=0)
    ks_p_value: Optional[float] = Field(None, ge=0, le=1)
    ad_statistic: Optional[float] = Field(None, ge=0)
    ad_p_value: Optional[float] = Field(None, ge=0, le=1)
    sw_statistic: Optional[float] = Field(None, ge=0, le=1)
    sw_p_value: Optional[float] = Field(None, ge=0, le=1)
    jb_statistic: Optional[float] = Field(None, ge=0)
    jb_p_value: Optional[float] = Field(None, ge=0, le=1)
    
    # Distribution parameters
    location: Optional[float] = None
    scale: Optional[float] = Field(None, gt=0)
    shape: Optional[list[float]] = None
    degrees_freedom: Optional[float] = Field(None, gt=0)
    
    # Model selection metrics
    log_likelihood: Optional[float] = None
    aic: Optional[float] = None
    bic: Optional[float] = None
    confidence_level: float = Field(default=0.95, gt=0, lt=1)
    parameter_confidence_intervals: Optional[dict[str, tuple[float, float]]] = None
    
    # Theoretical properties
    theoretical_mean: Optional[float] = None
    theoretical_variance: Optional[float] = Field(None, ge=0)
    theoretical_skewness: Optional[float] = None
    theoretical_kurtosis: Optional[float] = None
    
    # Empirical vs theoretical
    mean_difference: Optional[float] = None
    variance_ratio: Optional[float] = Field(None, gt=0)
    qq_plot_correlation: Optional[float] = Field(None, ge=-1, le=1)
    pp_plot_correlation: Optional[float] = Field(None, ge=-1, le=1)
    
    # Distribution assessment
    best_fit_test: Optional[DistributionTest] = None
    normality_p_value: Optional[float] = Field(None, ge=0, le=1)
    is_normal: Optional[bool] = None
    alternative_distributions: list[DistributionType] = Field(default_factory=list)
    
    # Tail properties
    left_tail_behavior: Optional[str] = None
    right_tail_behavior: Optional[str] = None
    tail_index: Optional[float] = None
    extreme_value_index: Optional[float] = None
    
    # Multivariate properties
    correlation_matrix: Optional[list[list[float]]] = None
    covariance_matrix: Optional[list[list[float]]] = None
    mahalanobis_distances: Optional[list[float]] = None
    eigenvalues: Optional[list[float]] = None
    condition_number: Optional[float] = Field(None, ge=1)
    
    # Transformation
    transformation_needed: bool = Field(default=False)
    recommended_transformation: Optional[str] = None
    transformation_lambda: Optional[float] = None
    transformed_distribution: Optional[DistributionType] = None
    
    # Advanced metrics
    entropy: Optional[float] = None
    information_content: Optional[float] = Field(None, ge=0)
    complexity_measure: Optional[float] = Field(None, ge=0)
    goodness_of_fit_score: Optional[float] = Field(None, ge=0, le=1)
    
    # Sampling properties
    random_seed: Optional[int] = None
    sample_statistics: Optional[dict[str, float]] = None
    bootstrap_confidence: Optional[dict[str, tuple[float, float]]] = None
    monte_carlo_samples: int = Field(default=10000, gt=0)
    
    @validator('parameters')
    def validate_parameters(cls, v: dict[str, float]) -> dict[str, float]:
        """Validate distribution parameters."""
        for param_name, param_value in v.items():
            if not isinstance(param_value, (int, float)):
                raise ValueError(f"Parameter {param_name} must be numeric")
        return v
    
    @validator('correlation_matrix')
    def validate_correlation_matrix(cls, v: Optional[list[list[float]]]) -> Optional[list[list[float]]]:
        """Validate correlation matrix properties."""
        if v is not None:
            n = len(v)
            if n == 0:
                return v
            
            # Check square matrix
            for row in v:
                if len(row) != n:
                    raise ValueError("Correlation matrix must be square")
            
            # Check diagonal elements are 1
            for i in range(n):
                if abs(v[i][i] - 1.0) > 1e-10:
                    raise ValueError("Correlation matrix diagonal elements must be 1")
            
            # Check values are in [-1, 1]
            for row in v:
                for val in row:
                    if not -1 <= val <= 1:
                        raise ValueError("Correlation values must be between -1 and 1")
        
        return v
    
    def get_distribution_summary(self) -> dict[str, Any]:
        """Get comprehensive distribution summary."""
        summary = {
            "sample_size": self.sample_size,
            "distribution_type": self.distribution_type.value,
            "parameters": self.parameters.copy()
        }
        
        # Add goodness of fit information
        gof_tests = {}
        if self.ks_p_value is not None:
            gof_tests["kolmogorov_smirnov"] = {
                "statistic": self.ks_statistic,
                "p_value": self.ks_p_value,
                "significant": self.ks_p_value < 0.05
            }
        
        if self.sw_p_value is not None:
            gof_tests["shapiro_wilk"] = {
                "statistic": self.sw_statistic,
                "p_value": self.sw_p_value,
                "significant": self.sw_p_value < 0.05
            }
        
        if gof_tests:
            summary["goodness_of_fit_tests"] = gof_tests
        
        # Add theoretical properties
        if self.theoretical_mean is not None:
            summary["theoretical_properties"] = {
                "mean": self.theoretical_mean,
                "variance": self.theoretical_variance,
                "skewness": self.theoretical_skewness,
                "kurtosis": self.theoretical_kurtosis
            }
        
        # Add assessment
        summary["assessment"] = {
            "is_normal": self.is_normal,
            "goodness_of_fit_score": self.goodness_of_fit_score,
            "transformation_needed": self.transformation_needed
        }
        
        return summary
    
    def test_normality(self, alpha: float = 0.05) -> dict[str, Any]:
        """Comprehensive normality testing."""
        tests = {}
        
        if self.sw_p_value is not None:
            tests["shapiro_wilk"] = {
                "statistic": self.sw_statistic,
                "p_value": self.sw_p_value,
                "is_normal": self.sw_p_value > alpha,
                "interpretation": "Normal" if self.sw_p_value > alpha else "Not normal"
            }
        
        if self.jb_p_value is not None:
            tests["jarque_bera"] = {
                "statistic": self.jb_statistic,
                "p_value": self.jb_p_value,
                "is_normal": self.jb_p_value > alpha,
                "interpretation": "Normal" if self.jb_p_value > alpha else "Not normal"
            }
        
        if self.ks_p_value is not None:
            tests["kolmogorov_smirnov"] = {
                "statistic": self.ks_statistic,
                "p_value": self.ks_p_value,
                "is_normal": self.ks_p_value > alpha,
                "interpretation": "Normal" if self.ks_p_value > alpha else "Not normal"
            }
        
        # Overall assessment
        normal_tests = [test["is_normal"] for test in tests.values() if "is_normal" in test]
        if normal_tests:
            overall_normal = sum(normal_tests) / len(normal_tests) > 0.5
            tests["overall_assessment"] = {
                "is_normal": overall_normal,
                "consensus": f"{sum(normal_tests)}/{len(normal_tests)} tests support normality",
                "recommendation": "Treat as normal" if overall_normal else "Consider non-normal distribution"
            }
        
        return tests
    
    def compare_with_normal(self) -> dict[str, Any]:
        """Compare this distribution with normal distribution."""
        comparison = {
            "distribution_type": self.distribution_type.value,
            "is_normal_distribution": self.distribution_type == DistributionType.NORMAL
        }
        
        if self.theoretical_mean is not None and self.theoretical_variance is not None:
            comparison["vs_normal"] = {
                "mean_difference": "N/A" if self.location is None else abs(self.theoretical_mean - self.location),
                "variance_ratio": "N/A" if self.scale is None else self.theoretical_variance / (self.scale**2),
                "skewness_difference": abs(self.theoretical_skewness) if self.theoretical_skewness is not None else "N/A",
                "excess_kurtosis": (self.theoretical_kurtosis - 3) if self.theoretical_kurtosis is not None else "N/A"
            }
        
        # Normality test results
        if self.normality_p_value is not None:
            comparison["normality_test"] = {
                "p_value": self.normality_p_value,
                "is_normal": self.is_normal,
                "confidence": "High" if abs(self.normality_p_value - 0.5) > 0.4 else "Medium"
            }
        
        return comparison
    
    def get_tail_analysis(self) -> dict[str, Any]:
        """Analyze tail behavior of the distribution."""
        tail_analysis = {
            "distribution_type": self.distribution_type.value
        }
        
        if self.left_tail_behavior:
            tail_analysis["left_tail"] = self.left_tail_behavior
        
        if self.right_tail_behavior:
            tail_analysis["right_tail"] = self.right_tail_behavior
        
        if self.tail_index is not None:
            tail_analysis["tail_index"] = self.tail_index
            if self.tail_index > 2:
                tail_analysis["tail_classification"] = "Heavy-tailed"
            elif self.tail_index > 1:
                tail_analysis["tail_classification"] = "Medium-tailed"
            else:
                tail_analysis["tail_classification"] = "Light-tailed"
        
        return tail_analysis
    
    def generate_samples(self, n_samples: int = 1000, 
                        random_state: Optional[int] = None) -> list[float]:
        """Generate random samples from the fitted distribution."""
        if random_state is None:
            random_state = self.random_seed
        
        import numpy as np
        
        if random_state is not None:
            np.random.seed(random_state)
        
        if self.distribution_type == DistributionType.NORMAL:
            mu = self.parameters.get("mu", self.location or 0)
            sigma = self.parameters.get("sigma", self.scale or 1)
            samples = np.random.normal(mu, sigma, n_samples)
        
        elif self.distribution_type == DistributionType.EXPONENTIAL:
            scale = self.parameters.get("lambda", 1)
            samples = np.random.exponential(1/scale, n_samples)
        
        elif self.distribution_type == DistributionType.UNIFORM:
            a = self.parameters.get("a", 0)
            b = self.parameters.get("b", 1)
            samples = np.random.uniform(a, b, n_samples)
        
        else:
            # For unsupported distributions, generate from normal as fallback
            mu = self.theoretical_mean or 0
            sigma = (self.theoretical_variance ** 0.5) if self.theoretical_variance else 1
            samples = np.random.normal(mu, sigma, n_samples)
        
        return samples.tolist()
    
    @classmethod
    def from_scipy_fit(cls, data: Any, distribution_name: str,
                      fitted_params: tuple, test_results: Optional[dict] = None,
                      **kwargs: Any) -> DataDistribution:
        """Create DataDistribution from scipy distribution fit."""
        import numpy as np
        from scipy import stats
        
        data_array = np.array(data)
        sample_size = len(data_array)
        
        # Map scipy distribution names to our enum
        dist_mapping = {
            "norm": DistributionType.NORMAL,
            "expon": DistributionType.EXPONENTIAL,
            "gamma": DistributionType.GAMMA,
            "uniform": DistributionType.UNIFORM,
            "beta": DistributionType.BETA,
            "lognorm": DistributionType.LOG_NORMAL,
            "chi2": DistributionType.CHI_SQUARED,
            "t": DistributionType.T_DISTRIBUTION,
            "weibull_min": DistributionType.WEIBULL,
            "pareto": DistributionType.PARETO,
            "laplace": DistributionType.LAPLACE
        }
        
        distribution_type = dist_mapping.get(distribution_name, DistributionType.UNKNOWN)
        
        # Extract parameters based on distribution type
        parameters = {}
        location = None
        scale = None
        
        if distribution_name == "norm":
            location, scale = fitted_params
            parameters = {"mu": location, "sigma": scale}
        elif distribution_name == "expon":
            location, scale = fitted_params
            parameters = {"lambda": 1/scale}
        elif distribution_name == "gamma":
            shape, location, scale = fitted_params
            parameters = {"alpha": shape, "beta": 1/scale}
        
        # Calculate theoretical moments
        try:
            scipy_dist = getattr(stats, distribution_name)(*fitted_params)
            theoretical_mean = float(scipy_dist.mean())
            theoretical_variance = float(scipy_dist.var())
            theoretical_skewness = float(scipy_dist.stats(moments='s'))
            theoretical_kurtosis = float(scipy_dist.stats(moments='k'))
        except:
            theoretical_mean = theoretical_variance = None
            theoretical_skewness = theoretical_kurtosis = None
        
        # Add test results if provided
        test_kwargs = {}
        if test_results:
            if "ks_test" in test_results:
                test_kwargs.update({
                    "ks_statistic": test_results["ks_test"][0],
                    "ks_p_value": test_results["ks_test"][1]
                })
            if "shapiro_test" in test_results:
                test_kwargs.update({
                    "sw_statistic": test_results["shapiro_test"][0],
                    "sw_p_value": test_results["shapiro_test"][1]
                })
        
        return cls(
            sample_size=sample_size,
            distribution_type=distribution_type,
            parameters=parameters,
            location=location,
            scale=scale,
            theoretical_mean=theoretical_mean,
            theoretical_variance=theoretical_variance,
            theoretical_skewness=theoretical_skewness,
            theoretical_kurtosis=theoretical_kurtosis,
            **test_kwargs,
            **kwargs
        )