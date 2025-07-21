# Statistics Package

A comprehensive statistical analysis and mathematical operations package for advanced data science and analytics.

## Overview

The Statistics package provides a robust foundation for statistical analysis, hypothesis testing, and mathematical operations. It offers both classical and modern statistical methods for data scientists, analysts, and researchers.

## Features

- **Descriptive Statistics**: Comprehensive descriptive statistical measures and summaries
- **Inferential Statistics**: Hypothesis testing, confidence intervals, and statistical inference
- **Probability Distributions**: Support for 50+ probability distributions with parameter estimation
- **Regression Analysis**: Linear, non-linear, and advanced regression modeling techniques
- **Time Series Analysis**: Specialized statistical methods for time series data
- **Bayesian Statistics**: Bayesian inference and probabilistic modeling
- **Multivariate Analysis**: Principal component analysis, factor analysis, clustering
- **Experimental Design**: A/B testing, ANOVA, and experimental design methodologies
- **Resampling Methods**: Bootstrap, permutation tests, and cross-validation
- **Statistical Visualization**: Advanced statistical plotting and diagnostic charts

## Architecture

```
statistics/
├── domain/                 # Core statistical business logic
│   ├── entities/          # Statistical models, tests, distributions
│   ├── services/          # Statistical computation services
│   └── value_objects/     # Statistical parameters and results
├── application/           # Use cases and orchestration  
│   ├── services/          # Application services
│   ├── use_cases/         # Statistical analysis workflows
│   └── dto/               # Data transfer objects
├── infrastructure/        # External integrations
│   ├── repositories/      # Statistical data storage
│   ├── adapters/          # R, SciPy, Stan adapters
│   └── computation/       # Numerical computation engines
└── presentation/          # Interfaces
    ├── api/               # REST API endpoints
    ├── cli/               # Command-line interface
    └── visualization/     # Statistical plots and charts
```

## Quick Start

```python
from src.packages.data.statistics.application.services import StatisticsService
from src.packages.data.statistics.domain.entities import Dataset, StatisticalTest

# Initialize statistics service
stats_service = StatisticsService()

# Descriptive statistics
dataset = Dataset.from_array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
descriptive_stats = stats_service.describe(dataset)

print(f"Mean: {descriptive_stats.mean}")
print(f"Standard Deviation: {descriptive_stats.std}")
print(f"Median: {descriptive_stats.median}")
print(f"Skewness: {descriptive_stats.skewness}")

# Hypothesis testing
sample_a = [23, 24, 25, 26, 27, 28, 29, 30]
sample_b = [20, 21, 22, 23, 24, 25, 26, 27]

t_test = stats_service.t_test(
    sample_a=sample_a,
    sample_b=sample_b,
    alternative="two-sided",
    alpha=0.05
)

print(f"T-statistic: {t_test.statistic}")
print(f"P-value: {t_test.p_value}")
print(f"Significant: {t_test.is_significant}")

# Regression analysis
regression = stats_service.linear_regression(
    x=dataset.values,
    y=[2*x + 1 + random.normal(0, 0.5) for x in dataset.values],
    include_intercept=True
)

print(f"R-squared: {regression.r_squared}")
print(f"Coefficients: {regression.coefficients}")
```

## Core Statistical Modules

### Descriptive Statistics
```python
# Comprehensive descriptive analysis
analysis = stats_service.comprehensive_analysis(dataset)
print(f"Five-number summary: {analysis.five_number_summary}")
print(f"Distribution shape: {analysis.distribution_shape}")
print(f"Outliers detected: {len(analysis.outliers)}")

# Custom statistics
custom_stats = stats_service.calculate_custom_statistics(
    dataset,
    statistics=["geometric_mean", "harmonic_mean", "trimmed_mean"]
)
```

### Probability Distributions
```python
from src.packages.data.statistics.domain.entities import Distribution

# Fit distribution to data
best_fit = stats_service.fit_distribution(
    data=dataset,
    distributions=["normal", "lognormal", "gamma", "weibull"]
)

print(f"Best fitting distribution: {best_fit.distribution_name}")
print(f"Parameters: {best_fit.parameters}")
print(f"Goodness of fit: {best_fit.goodness_of_fit}")

# Generate samples from distribution
normal_dist = Distribution.normal(mean=100, std=15)
samples = stats_service.generate_samples(normal_dist, n_samples=1000)
```

### Hypothesis Testing
```python
# Multiple comparison procedures
anova_result = stats_service.one_way_anova(
    groups=[group_a, group_b, group_c],
    post_hoc="tukey"
)

# Non-parametric tests
mann_whitney = stats_service.mann_whitney_u_test(sample_a, sample_b)
wilcoxon = stats_service.wilcoxon_signed_rank_test(before, after)

# Goodness of fit tests
ks_test = stats_service.kolmogorov_smirnov_test(data, "normal")
shapiro_test = stats_service.shapiro_wilk_test(data)
```

### Regression Analysis
```python
# Multiple regression
multiple_reg = stats_service.multiple_regression(
    y=dependent_variable,
    X=independent_variables,
    method="ols"
)

# Logistic regression
logistic_reg = stats_service.logistic_regression(
    y=binary_outcome,
    X=predictors,
    regularization="l2"
)

# Robust regression
robust_reg = stats_service.robust_regression(
    y=dependent_variable,
    X=independent_variables,
    method="huber"
)
```

### Time Series Analysis
```python
# Time series decomposition
decomposition = stats_service.decompose_time_series(
    data=time_series_data,
    model="additive",
    period=12
)

# Stationarity testing
adf_test = stats_service.augmented_dickey_fuller_test(time_series_data)
kpss_test = stats_service.kpss_test(time_series_data)

# ARIMA modeling
arima_model = stats_service.fit_arima(
    data=time_series_data,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12)
)
```

### Bayesian Statistics
```python
# Bayesian parameter estimation
bayesian_result = stats_service.bayesian_estimation(
    data=sample_data,
    prior_distribution="normal",
    prior_parameters={"mean": 0, "std": 1}
)

# Markov Chain Monte Carlo
mcmc_samples = stats_service.mcmc_sampling(
    model=bayesian_model,
    n_samples=10000,
    n_chains=4
)

# Bayesian A/B testing
ab_test_bayesian = stats_service.bayesian_ab_test(
    control_group=control_data,
    treatment_group=treatment_data,
    metric="conversion_rate"
)
```

## Use Cases

- **A/B Testing**: Statistical significance testing for experiments
- **Quality Control**: Statistical process control and quality assurance
- **Risk Analysis**: Statistical risk modeling and assessment
- **Market Research**: Survey analysis and market research statistics
- **Scientific Research**: Experimental design and statistical validation
- **Financial Analysis**: Statistical analysis of financial data and risk metrics
- **Manufacturing**: Statistical quality control and process optimization

## Integration

Works with other data domain packages:

```python
# With data quality for statistical validation
from src.packages.data.data_quality.application.services import DataQualityService

quality_service = DataQualityService()
quality_report = quality_service.assess_data_quality(dataset)

if quality_report.is_statistically_valid:
    statistical_analysis = stats_service.comprehensive_analysis(dataset)

# With data analytics for enhanced insights
from src.packages.data.data_analytics.application.services import AnalyticsService

analytics = AnalyticsService()
analytics.add_statistical_insights(statistical_analysis)
```

## Installation

```bash
# Install from package directory
cd src/packages/data/statistics
pip install -e .

# Install with advanced statistical dependencies
pip install -e ".[scipy,statsmodels,pymc]"
```

## Configuration

```yaml
# statistics_config.yaml
statistics:
  computation:
    precision: "double"
    random_seed: 42
    parallel_processing: true
    
  testing:
    default_alpha: 0.05
    multiple_comparisons: "bonferroni"
    effect_size_calculation: true
    
  bayesian:
    default_chains: 4
    default_samples: 10000
    convergence_diagnostics: true
    
  visualization:
    style: "seaborn"
    color_palette: "colorblind"
    figure_size: [10, 6]
```

## Performance

Optimized for large-scale statistical computation:

- **Vectorized Operations**: NumPy and SciPy optimized computations
- **Parallel Processing**: Multi-core statistical computations
- **Memory Efficiency**: Streaming algorithms for large datasets
- **Numerical Stability**: Robust numerical algorithms
- **Caching**: Intelligent caching of computation-intensive operations

## License

MIT License