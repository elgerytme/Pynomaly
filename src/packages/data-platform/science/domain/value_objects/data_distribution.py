"""Data Distribution value object for statistical distribution analysis."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import Field, validator

from packages.core.domain.abstractions.base_value_object import BaseValueObject


class DataDistribution(BaseValueObject):
    """Value object representing statistical distribution characteristics.
    
    This immutable value object encapsulates the statistical distribution
    properties of numerical data including distribution type, parameters,
    and goodness-of-fit measures.
    
    Attributes:
        distribution_name: Name of the fitted distribution
        parameters: Distribution parameters (e.g., mean, std for normal)
        goodness_of_fit: Goodness-of-fit score (0-1, higher is better)
        test_statistic: Statistical test value
        p_value: P-value from goodness-of-fit test
        confidence_interval: Confidence interval for parameters
        sample_size: Number of observations used for fitting
        log_likelihood: Log-likelihood of the fitted distribution
        aic_score: Akaike Information Criterion score
        bic_score: Bayesian Information Criterion score
        ks_statistic: Kolmogorov-Smirnov test statistic
        ks_p_value: Kolmogorov-Smirnov test p-value
        anderson_statistic: Anderson-Darling test statistic
        anderson_critical_values: Critical values for Anderson-Darling test
        shapiro_statistic: Shapiro-Wilk test statistic (for normality)
        shapiro_p_value: Shapiro-Wilk test p-value
        jarque_bera_statistic: Jarque-Bera test statistic
        jarque_bera_p_value: Jarque-Bera test p-value
        outlier_detection: Outlier information based on distribution
        is_significant_fit: Whether the distribution fit is statistically significant
        alternative_distributions: Other distributions that were tested
    """
    
    # Primary distribution characteristics
    distribution_name: str = Field(..., min_length=1)
    parameters: dict[str, float] = Field(default_factory=dict)
    goodness_of_fit: float = Field(..., ge=0, le=1)
    test_statistic: Optional[float] = None
    p_value: Optional[float] = Field(None, ge=0, le=1)
    
    # Confidence and likelihood
    confidence_interval: Optional[dict[str, tuple[float, float]]] = None
    sample_size: int = Field(..., gt=0)
    log_likelihood: Optional[float] = None
    
    # Information criteria
    aic_score: Optional[float] = None
    bic_score: Optional[float] = None
    
    # Specific distribution tests
    ks_statistic: Optional[float] = Field(None, ge=0)
    ks_p_value: Optional[float] = Field(None, ge=0, le=1)
    anderson_statistic: Optional[float] = None
    anderson_critical_values: Optional[list[float]] = None
    shapiro_statistic: Optional[float] = None
    shapiro_p_value: Optional[float] = Field(None, ge=0, le=1)
    jarque_bera_statistic: Optional[float] = None
    jarque_bera_p_value: Optional[float] = Field(None, ge=0, le=1)
    
    # Additional analysis
    outlier_detection: dict[str, Any] = Field(default_factory=dict)
    is_significant_fit: Optional[bool] = None
    alternative_distributions: list[dict[str, Any]] = Field(default_factory=list)
    
    @validator('distribution_name')
    def validate_distribution_name(cls, v: str) -> str:
        """Validate distribution name."""
        valid_distributions = {
            'normal', 'lognormal', 'exponential', 'gamma', 'beta', 'uniform',
            'binomial', 'poisson', 'geometric', 'chi2', 'f', 't', 'weibull',
            'pareto', 'laplace', 'cauchy', 'logistic', 'gumbel'
        }
        
        if v.lower() not in valid_distributions:
            # Allow custom distributions but warn
            pass
            
        return v
    
    @validator('parameters')
    def validate_parameters(cls, v: dict[str, float]) -> dict[str, float]:
        """Validate distribution parameters."""
        if not v:
            raise ValueError("Distribution parameters cannot be empty")
            
        for param_name, param_value in v.items():
            if not isinstance(param_value, (int, float)):
                raise ValueError(f"Parameter '{param_name}' must be numeric")
                
        return v
    
    @validator('is_significant_fit')
    def validate_significance(cls, v: Optional[bool], values: dict[str, Any]) -> Optional[bool]:
        """Validate significance based on p-value."""
        if v is None:
            p_value = values.get('p_value')
            if p_value is not None:
                return p_value > 0.05  # Common significance threshold
        return v
    
    def get_distribution_family(self) -> str:
        """Get the family of the distribution."""
        continuous_distributions = {
            'normal', 'lognormal', 'exponential', 'gamma', 'beta', 'uniform',
            'chi2', 'f', 't', 'weibull', 'pareto', 'laplace', 'cauchy', 
            'logistic', 'gumbel'
        }
        
        discrete_distributions = {
            'binomial', 'poisson', 'geometric', 'negative_binomial'
        }
        
        dist_name = self.distribution_name.lower()
        
        if dist_name in continuous_distributions:
            return "continuous"
        elif dist_name in discrete_distributions:
            return "discrete"
        else:
            return "unknown"
    
    def is_good_fit(self, threshold: float = 0.7) -> bool:
        """Check if the distribution is a good fit based on goodness-of-fit score."""
        return self.goodness_of_fit >= threshold
    
    def is_normal_distribution(self) -> bool:
        """Check if this represents a normal distribution."""
        return self.distribution_name.lower() in ['normal', 'gaussian']
    
    def get_parameter_value(self, parameter_name: str) -> Optional[float]:
        """Get a specific parameter value."""
        return self.parameters.get(parameter_name)
    
    def get_mean(self) -> Optional[float]:
        """Get the mean of the distribution if available."""
        if self.is_normal_distribution():
            return self.parameters.get('loc') or self.parameters.get('mean')
        elif self.distribution_name.lower() == 'exponential':
            scale = self.parameters.get('scale', 1.0)
            return scale
        elif self.distribution_name.lower() == 'gamma':
            shape = self.parameters.get('a') or self.parameters.get('shape')
            scale = self.parameters.get('scale', 1.0)
            if shape:
                return shape * scale
        return None
    
    def get_variance(self) -> Optional[float]:
        """Get the variance of the distribution if available."""
        if self.is_normal_distribution():
            std = self.parameters.get('scale') or self.parameters.get('std')
            return std ** 2 if std else None
        elif self.distribution_name.lower() == 'exponential':
            scale = self.parameters.get('scale', 1.0)
            return scale ** 2
        elif self.distribution_name.lower() == 'gamma':
            shape = self.parameters.get('a') or self.parameters.get('shape')
            scale = self.parameters.get('scale', 1.0)
            if shape:
                return shape * (scale ** 2)
        return None
    
    def get_confidence_interval_for_parameter(self, parameter_name: str) -> Optional[tuple[float, float]]:
        """Get confidence interval for a specific parameter."""
        if not self.confidence_interval:
            return None
        return self.confidence_interval.get(parameter_name)
    
    def compare_information_criteria(self, other: DataDistribution) -> dict[str, str]:
        """Compare information criteria with another distribution."""
        if not isinstance(other, DataDistribution):
            raise ValueError("Can only compare with another DataDistribution")
        
        comparison = {}
        
        if self.aic_score is not None and other.aic_score is not None:
            if self.aic_score < other.aic_score:
                comparison["aic_winner"] = f"{self.distribution_name} (lower AIC)"
            elif other.aic_score < self.aic_score:
                comparison["aic_winner"] = f"{other.distribution_name} (lower AIC)"
            else:
                comparison["aic_winner"] = "tie"
        
        if self.bic_score is not None and other.bic_score is not None:
            if self.bic_score < other.bic_score:
                comparison["bic_winner"] = f"{self.distribution_name} (lower BIC)"
            elif other.bic_score < self.bic_score:
                comparison["bic_winner"] = f"{other.distribution_name} (lower BIC)"
            else:
                comparison["bic_winner"] = "tie"
        
        if self.goodness_of_fit > other.goodness_of_fit:
            comparison["fit_winner"] = f"{self.distribution_name} (better fit)"
        elif other.goodness_of_fit > self.goodness_of_fit:
            comparison["fit_winner"] = f"{other.distribution_name} (better fit)"
        else:
            comparison["fit_winner"] = "tie"
        
        return comparison
    
    def get_outlier_bounds(self) -> Optional[tuple[float, float]]:
        """Get outlier bounds based on the distribution."""
        if not self.outlier_detection:
            return None
            
        lower_bound = self.outlier_detection.get('lower_bound')
        upper_bound = self.outlier_detection.get('upper_bound')
        
        if lower_bound is not None and upper_bound is not None:
            return (lower_bound, upper_bound)
        
        return None
    
    def get_distribution_summary(self) -> dict[str, Any]:
        """Get a comprehensive summary of the distribution."""
        summary = {
            "distribution": self.distribution_name,
            "family": self.get_distribution_family(),
            "sample_size": self.sample_size,
            "goodness_of_fit": self.goodness_of_fit,
            "is_good_fit": self.is_good_fit(),
            "is_significant": self.is_significant_fit,
            "parameters": self.parameters,
        }
        
        # Add derived statistics if available
        mean = self.get_mean()
        variance = self.get_variance()
        
        if mean is not None:
            summary["mean"] = mean
        if variance is not None:
            summary["variance"] = variance
            summary["std"] = variance ** 0.5
        
        # Add test results
        test_results = {}
        if self.ks_p_value is not None:
            test_results["kolmogorov_smirnov"] = {
                "statistic": self.ks_statistic,
                "p_value": self.ks_p_value
            }
        
        if self.shapiro_p_value is not None:
            test_results["shapiro_wilk"] = {
                "statistic": self.shapiro_statistic,
                "p_value": self.shapiro_p_value
            }
        
        if self.jarque_bera_p_value is not None:
            test_results["jarque_bera"] = {
                "statistic": self.jarque_bera_statistic,
                "p_value": self.jarque_bera_p_value
            }
        
        if test_results:
            summary["test_results"] = test_results
        
        # Add information criteria if available
        if self.aic_score is not None or self.bic_score is not None:
            summary["information_criteria"] = {
                "aic": self.aic_score,
                "bic": self.bic_score,
                "log_likelihood": self.log_likelihood
            }
        
        return summary
    
    @classmethod
    def from_scipy_fit(cls, distribution_name: str, parameters: dict[str, float],
                      data: Any, **kwargs: Any) -> DataDistribution:
        """Create DataDistribution from scipy distribution fit."""
        import numpy as np
        from scipy import stats
        
        # Convert data to numpy array
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        # Get distribution object
        dist = getattr(stats, distribution_name, None)
        if dist is None:
            raise ValueError(f"Unknown distribution: {distribution_name}")
        
        # Calculate goodness of fit using Kolmogorov-Smirnov test
        ks_stat, ks_p = stats.kstest(data, dist.cdf, args=tuple(parameters.values()))
        
        # Calculate information criteria
        log_likelihood = np.sum(dist.logpdf(data, *parameters.values()))
        k = len(parameters)  # number of parameters
        n = len(data)  # sample size
        
        aic = 2 * k - 2 * log_likelihood
        bic = k * np.log(n) - 2 * log_likelihood
        
        # Goodness of fit score (1 - normalized KS statistic)
        goodness_of_fit = max(0, 1 - ks_stat)
        
        return cls(
            distribution_name=distribution_name,
            parameters=parameters,
            goodness_of_fit=goodness_of_fit,
            test_statistic=ks_stat,
            p_value=ks_p,
            sample_size=len(data),
            log_likelihood=log_likelihood,
            aic_score=aic,
            bic_score=bic,
            ks_statistic=ks_stat,
            ks_p_value=ks_p,
            is_significant_fit=ks_p > 0.05,
            **kwargs
        )