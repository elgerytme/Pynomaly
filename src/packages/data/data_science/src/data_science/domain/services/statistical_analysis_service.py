"""Statistical Analysis Service for comprehensive data analysis."""

from __future__ import annotations

from typing import Dict, List, Optional, Any, Tuple, Union
import math
import statistics
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy import stats
from scipy.stats import normaltest, shapiro, kstest, chi2_contingency


class StatisticalTest(Enum):
    """Types of statistical tests available."""
    
    T_TEST_ONE_SAMPLE = "t_test_one_sample"
    T_TEST_TWO_SAMPLE = "t_test_two_sample"
    T_TEST_PAIRED = "t_test_paired"
    WELCH_T_TEST = "welch_t_test"
    MANN_WHITNEY_U = "mann_whitney_u"
    WILCOXON_SIGNED_RANK = "wilcoxon_signed_rank"
    KOLMOGOROV_SMIRNOV = "kolmogorov_smirnov"
    SHAPIRO_WILK = "shapiro_wilk"
    ANDERSON_DARLING = "anderson_darling"
    CHI_SQUARE = "chi_square"
    FISHER_EXACT = "fisher_exact"
    ANOVA_ONE_WAY = "anova_one_way"
    KRUSKAL_WALLIS = "kruskal_wallis"
    PEARSON_CORRELATION = "pearson_correlation"
    SPEARMAN_CORRELATION = "spearman_correlation"
    KENDALL_TAU = "kendall_tau"


class DistributionType(Enum):
    """Types of probability distributions."""
    
    NORMAL = "normal"
    UNIFORM = "uniform"
    EXPONENTIAL = "exponential"
    POISSON = "poisson"
    BINOMIAL = "binomial"
    GAMMA = "gamma"
    BETA = "beta"
    CHI_SQUARE = "chi_square"
    STUDENT_T = "student_t"
    F_DISTRIBUTION = "f_distribution"


@dataclass
class StatisticalTestResult:
    """Result of a statistical test."""
    
    test_type: StatisticalTest
    statistic: float
    p_value: float
    critical_value: Optional[float] = None
    confidence_level: float = 0.95
    degrees_of_freedom: Optional[int] = None
    effect_size: Optional[float] = None
    power: Optional[float] = None
    
    # Interpretation
    is_significant: bool = False
    interpretation: str = ""
    
    # Additional details
    alternative_hypothesis: str = "two-sided"
    method_used: str = ""
    assumptions_met: Dict[str, bool] = None
    
    # Metadata
    sample_sizes: List[int] = None
    test_timestamp: datetime = None
    
    def __post_init__(self):
        """Initialize computed fields."""
        if self.test_timestamp is None:
            self.test_timestamp = datetime.utcnow()
        
        if self.assumptions_met is None:
            self.assumptions_met = {}
        
        if self.sample_sizes is None:
            self.sample_sizes = []
        
        # Determine significance
        alpha = 1 - self.confidence_level
        self.is_significant = self.p_value < alpha
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "test_type": self.test_type.value,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "critical_value": self.critical_value,
            "confidence_level": self.confidence_level,
            "degrees_of_freedom": self.degrees_of_freedom,
            "effect_size": self.effect_size,
            "power": self.power,
            "is_significant": self.is_significant,
            "interpretation": self.interpretation,
            "alternative_hypothesis": self.alternative_hypothesis,
            "method_used": self.method_used,
            "assumptions_met": self.assumptions_met,
            "sample_sizes": self.sample_sizes,
            "test_timestamp": self.test_timestamp.isoformat() if self.test_timestamp else None
        }


@dataclass
class DistributionAnalysis:
    """Analysis of data distribution."""
    
    distribution_type: Optional[DistributionType] = None
    parameters: Dict[str, float] = None
    goodness_of_fit_p_value: Optional[float] = None
    
    # Descriptive statistics
    mean: float = 0.0
    median: float = 0.0
    mode: Optional[float] = None
    variance: float = 0.0
    standard_deviation: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    
    # Quantiles
    quantiles: Dict[str, float] = None
    
    # Range statistics
    min_value: float = 0.0
    max_value: float = 0.0
    range_value: float = 0.0
    interquartile_range: float = 0.0
    
    # Normality tests
    normality_tests: Dict[str, StatisticalTestResult] = None
    
    def __post_init__(self):
        """Initialize default fields."""
        if self.parameters is None:
            self.parameters = {}
        
        if self.quantiles is None:
            self.quantiles = {}
        
        if self.normality_tests is None:
            self.normality_tests = {}


class StatisticalAnalysisService:
    """Service for comprehensive statistical analysis and testing."""
    
    def __init__(self, default_confidence_level: float = 0.95):
        """Initialize the statistical analysis service."""
        self.default_confidence_level = default_confidence_level
    
    def compute_descriptive_statistics(self, data: List[float]) -> Dict[str, float]:
        """Compute comprehensive descriptive statistics."""
        if not data:
            raise ValueError("Data cannot be empty")
        
        data_array = np.array(data)
        
        # Basic statistics
        stats_dict = {
            "count": len(data),
            "mean": float(np.mean(data_array)),
            "median": float(np.median(data_array)),
            "variance": float(np.var(data_array, ddof=1)) if len(data) > 1 else 0.0,
            "standard_deviation": float(np.std(data_array, ddof=1)) if len(data) > 1 else 0.0,
            "min": float(np.min(data_array)),
            "max": float(np.max(data_array)),
            "range": float(np.ptp(data_array)),
        }
        
        # Additional statistics
        if len(data) > 1:
            stats_dict.update({
                "skewness": float(stats.skew(data_array)),
                "kurtosis": float(stats.kurtosis(data_array)),
                "standard_error": stats_dict["standard_deviation"] / math.sqrt(len(data))
            })
        
        # Quantiles
        percentiles = [5, 10, 25, 50, 75, 90, 95]
        for p in percentiles:
            stats_dict[f"percentile_{p}"] = float(np.percentile(data_array, p))
        
        stats_dict["interquartile_range"] = stats_dict["percentile_75"] - stats_dict["percentile_25"]
        
        # Mode (handling potential multiple modes)
        try:
            mode_result = stats.mode(data_array, keepdims=False)
            if hasattr(mode_result, 'mode'):
                stats_dict["mode"] = float(mode_result.mode)
                stats_dict["mode_count"] = int(mode_result.count)
            else:
                stats_dict["mode"] = float(mode_result[0])
                stats_dict["mode_count"] = int(mode_result[1])
        except Exception:
            stats_dict["mode"] = None
            stats_dict["mode_count"] = 0
        
        return stats_dict
    
    def analyze_distribution(self, data: List[float]) -> DistributionAnalysis:
        """Perform comprehensive distribution analysis."""
        if not data or len(data) < 3:
            raise ValueError("Need at least 3 data points for distribution analysis")
        
        data_array = np.array(data)
        
        # Compute descriptive statistics
        desc_stats = self.compute_descriptive_statistics(data)
        
        # Create analysis object
        analysis = DistributionAnalysis(
            mean=desc_stats["mean"],
            median=desc_stats["median"],
            mode=desc_stats.get("mode"),
            variance=desc_stats["variance"],
            standard_deviation=desc_stats["standard_deviation"],
            skewness=desc_stats["skewness"],
            kurtosis=desc_stats["kurtosis"],
            min_value=desc_stats["min"],
            max_value=desc_stats["max"],
            range_value=desc_stats["range"],
            interquartile_range=desc_stats["interquartile_range"]
        )
        
        # Quantiles
        analysis.quantiles = {
            "q1": desc_stats["percentile_25"],
            "q2": desc_stats["percentile_50"],
            "q3": desc_stats["percentile_75"],
            "p5": desc_stats["percentile_5"],
            "p95": desc_stats["percentile_95"]
        }
        
        # Normality tests
        if len(data) >= 8:  # Minimum sample size for reliable normality tests
            analysis.normality_tests = self._perform_normality_tests(data_array)
        
        # Distribution fitting
        analysis.distribution_type, analysis.parameters = self._fit_best_distribution(data_array)
        
        return analysis
    
    def _perform_normality_tests(self, data: np.ndarray) -> Dict[str, StatisticalTestResult]:
        """Perform various normality tests."""
        normality_tests = {}
        
        # Shapiro-Wilk test (best for small samples)
        if len(data) <= 5000:  # Shapiro-Wilk works best for smaller samples
            try:
                statistic, p_value = shapiro(data)
                normality_tests["shapiro_wilk"] = StatisticalTestResult(
                    test_type=StatisticalTest.SHAPIRO_WILK,
                    statistic=statistic,
                    p_value=p_value,
                    interpretation="Normal" if p_value > 0.05 else "Not normal",
                    method_used="Shapiro-Wilk normality test"
                )
            except Exception as e:
                pass
        
        # D'Agostino's normality test
        if len(data) >= 8:
            try:
                statistic, p_value = normaltest(data)
                normality_tests["dagostino"] = StatisticalTestResult(
                    test_type=StatisticalTest.CHI_SQUARE,  # D'Agostino uses chi-square
                    statistic=statistic,
                    p_value=p_value,
                    interpretation="Normal" if p_value > 0.05 else "Not normal",
                    method_used="D'Agostino's normality test"
                )
            except Exception:
                pass
        
        # Kolmogorov-Smirnov test against normal distribution
        try:
            # Standardize data
            standardized = (data - np.mean(data)) / np.std(data)
            statistic, p_value = kstest(standardized, 'norm')
            normality_tests["kolmogorov_smirnov"] = StatisticalTestResult(
                test_type=StatisticalTest.KOLMOGOROV_SMIRNOV,
                statistic=statistic,
                p_value=p_value,
                interpretation="Normal" if p_value > 0.05 else "Not normal",
                method_used="Kolmogorov-Smirnov test against normal distribution"
            )
        except Exception:
            pass
        
        return normality_tests
    
    def _fit_best_distribution(self, data: np.ndarray) -> Tuple[Optional[DistributionType], Dict[str, float]]:
        """Fit data to various distributions and return the best fit."""
        if len(data) < 10:
            return None, {}
        
        # Common distributions to test
        distributions = [
            (DistributionType.NORMAL, stats.norm),
            (DistributionType.EXPONENTIAL, stats.expon),
            (DistributionType.UNIFORM, stats.uniform),
            (DistributionType.GAMMA, stats.gamma),
            (DistributionType.BETA, stats.beta)
        ]
        
        best_dist = None
        best_params = {}
        best_ks_stat = np.inf
        
        for dist_type, dist_obj in distributions:
            try:
                # Fit distribution
                params = dist_obj.fit(data)
                
                # Perform Kolmogorov-Smirnov test
                ks_stat, p_value = stats.kstest(data, lambda x: dist_obj.cdf(x, *params))
                
                # Keep track of best fit (lowest KS statistic)
                if ks_stat < best_ks_stat:
                    best_ks_stat = ks_stat
                    best_dist = dist_type
                    
                    # Create parameter dictionary
                    param_names = dist_obj.shapes.split(',') if dist_obj.shapes else []
                    param_names = [name.strip() for name in param_names] + ['loc', 'scale']
                    
                    best_params = {
                        name: float(param) for name, param in zip(param_names, params)
                        if not np.isnan(param)
                    }
                    best_params["ks_statistic"] = ks_stat
                    best_params["ks_p_value"] = p_value
                    
            except Exception:
                continue
        
        return best_dist, best_params
    
    def perform_t_test(
        self,
        data1: List[float],
        data2: Optional[List[float]] = None,
        population_mean: Optional[float] = None,
        test_type: str = "two_sample",
        alternative: str = "two-sided",
        confidence_level: float = None
    ) -> StatisticalTestResult:
        """Perform various types of t-tests."""
        if confidence_level is None:
            confidence_level = self.default_confidence_level
        
        data1_array = np.array(data1)
        
        if test_type == "one_sample":
            if population_mean is None:
                raise ValueError("Population mean required for one-sample t-test")
            
            statistic, p_value = stats.ttest_1samp(data1_array, population_mean, alternative=alternative)
            df = len(data1) - 1
            test_enum = StatisticalTest.T_TEST_ONE_SAMPLE
            
        elif test_type == "two_sample":
            if data2 is None:
                raise ValueError("Second data set required for two-sample t-test")
            
            data2_array = np.array(data2)
            statistic, p_value = stats.ttest_ind(data1_array, data2_array, alternative=alternative)
            df = len(data1) + len(data2) - 2
            test_enum = StatisticalTest.T_TEST_TWO_SAMPLE
            
        elif test_type == "paired":
            if data2 is None:
                raise ValueError("Second data set required for paired t-test")
            
            data2_array = np.array(data2)
            if len(data1) != len(data2):
                raise ValueError("Data sets must have same length for paired t-test")
            
            statistic, p_value = stats.ttest_rel(data1_array, data2_array, alternative=alternative)
            df = len(data1) - 1
            test_enum = StatisticalTest.T_TEST_PAIRED
            
        elif test_type == "welch":
            if data2 is None:
                raise ValueError("Second data set required for Welch's t-test")
            
            data2_array = np.array(data2)
            statistic, p_value = stats.ttest_ind(data1_array, data2_array, equal_var=False, alternative=alternative)
            # Welch's t-test degrees of freedom calculation
            s1, s2 = np.var(data1_array, ddof=1), np.var(data2_array, ddof=1)
            n1, n2 = len(data1), len(data2)
            df = (s1/n1 + s2/n2)**2 / ((s1/n1)**2/(n1-1) + (s2/n2)**2/(n2-1))
            test_enum = StatisticalTest.WELCH_T_TEST
            
        else:
            raise ValueError(f"Unknown t-test type: {test_type}")
        
        # Calculate critical value
        alpha = 1 - confidence_level
        if alternative == "two-sided":
            critical_value = stats.t.ppf(1 - alpha/2, df)
        else:
            critical_value = stats.t.ppf(1 - alpha, df)
        
        # Calculate effect size (Cohen's d)
        effect_size = None
        if test_type in ["two_sample", "welch"] and data2 is not None:
            pooled_std = np.sqrt(((len(data1)-1)*np.var(data1_array, ddof=1) + 
                                 (len(data2)-1)*np.var(data2_array, ddof=1)) / 
                                (len(data1) + len(data2) - 2))
            effect_size = abs(np.mean(data1_array) - np.mean(data2_array)) / pooled_std
        
        return StatisticalTestResult(
            test_type=test_enum,
            statistic=statistic,
            p_value=p_value,
            critical_value=critical_value,
            confidence_level=confidence_level,
            degrees_of_freedom=int(df),
            effect_size=effect_size,
            alternative_hypothesis=alternative,
            method_used=f"{test_type.replace('_', ' ').title()} T-test",
            sample_sizes=[len(data1)] + ([len(data2)] if data2 else []),
            interpretation=f"{'Significant' if p_value < (1-confidence_level) else 'Not significant'} at {confidence_level*100}% confidence level"
        )
    
    def perform_correlation_analysis(
        self,
        x_data: List[float],
        y_data: List[float],
        method: str = "pearson"
    ) -> StatisticalTestResult:
        """Perform correlation analysis between two variables."""
        if len(x_data) != len(y_data):
            raise ValueError("Data sets must have the same length")
        
        if len(x_data) < 3:
            raise ValueError("Need at least 3 data points for correlation analysis")
        
        x_array = np.array(x_data)
        y_array = np.array(y_data)
        
        if method == "pearson":
            correlation, p_value = stats.pearsonr(x_array, y_array)
            test_type = StatisticalTest.PEARSON_CORRELATION
            method_name = "Pearson correlation"
        elif method == "spearman":
            correlation, p_value = stats.spearmanr(x_array, y_array)
            test_type = StatisticalTest.SPEARMAN_CORRELATION
            method_name = "Spearman rank correlation"
        elif method == "kendall":
            correlation, p_value = stats.kendalltau(x_array, y_array)
            test_type = StatisticalTest.KENDALL_TAU
            method_name = "Kendall's tau correlation"
        else:
            raise ValueError(f"Unknown correlation method: {method}")
        
        # Interpret correlation strength
        abs_corr = abs(correlation)
        if abs_corr >= 0.7:
            strength = "strong"
        elif abs_corr >= 0.5:
            strength = "moderate"
        elif abs_corr >= 0.3:
            strength = "weak"
        else:
            strength = "very weak"
        
        interpretation = f"{strength.title()} {'positive' if correlation > 0 else 'negative'} correlation (r = {correlation:.3f})"
        
        return StatisticalTestResult(
            test_type=test_type,
            statistic=correlation,
            p_value=p_value,
            confidence_level=self.default_confidence_level,
            degrees_of_freedom=len(x_data) - 2,
            method_used=method_name,
            sample_sizes=[len(x_data)],
            interpretation=interpretation
        )
    
    def detect_outliers(
        self,
        data: List[float],
        method: str = "iqr",
        threshold: float = 1.5
    ) -> Dict[str, Any]:
        """Detect outliers in the data using various methods."""
        data_array = np.array(data)
        
        outliers = {}
        outlier_indices = []
        
        if method == "iqr":
            # Interquartile Range method
            q1 = np.percentile(data_array, 25)
            q3 = np.percentile(data_array, 75)
            iqr = q3 - q1
            
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            
            outlier_mask = (data_array < lower_bound) | (data_array > upper_bound)
            outlier_indices = np.where(outlier_mask)[0].tolist()
            
            outliers = {
                "method": "IQR",
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "threshold": threshold,
                "outlier_indices": outlier_indices,
                "outlier_values": data_array[outlier_mask].tolist(),
                "outlier_count": len(outlier_indices),
                "outlier_percentage": len(outlier_indices) / len(data) * 100
            }
            
        elif method == "zscore":
            # Z-score method
            mean = np.mean(data_array)
            std = np.std(data_array)
            z_scores = np.abs((data_array - mean) / std)
            
            outlier_mask = z_scores > threshold
            outlier_indices = np.where(outlier_mask)[0].tolist()
            
            outliers = {
                "method": "Z-Score",
                "threshold": threshold,
                "mean": mean,
                "std": std,
                "outlier_indices": outlier_indices,
                "outlier_values": data_array[outlier_mask].tolist(),
                "outlier_z_scores": z_scores[outlier_mask].tolist(),
                "outlier_count": len(outlier_indices),
                "outlier_percentage": len(outlier_indices) / len(data) * 100
            }
            
        elif method == "modified_zscore":
            # Modified Z-score using median absolute deviation
            median = np.median(data_array)
            mad = np.median(np.abs(data_array - median))
            modified_z_scores = 0.6745 * (data_array - median) / mad
            
            outlier_mask = np.abs(modified_z_scores) > threshold
            outlier_indices = np.where(outlier_mask)[0].tolist()
            
            outliers = {
                "method": "Modified Z-Score",
                "threshold": threshold,
                "median": median,
                "mad": mad,
                "outlier_indices": outlier_indices,
                "outlier_values": data_array[outlier_mask].tolist(),
                "outlier_modified_z_scores": modified_z_scores[outlier_mask].tolist(),
                "outlier_count": len(outlier_indices),
                "outlier_percentage": len(outlier_indices) / len(data) * 100
            }
            
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
        
        return outliers
    
    def calculate_confidence_interval(
        self,
        data: List[float],
        confidence_level: float = None,
        method: str = "t"
    ) -> Dict[str, float]:
        """Calculate confidence interval for the mean."""
        if confidence_level is None:
            confidence_level = self.default_confidence_level
        
        data_array = np.array(data)
        n = len(data_array)
        mean = np.mean(data_array)
        std_err = stats.sem(data_array)
        
        alpha = 1 - confidence_level
        
        if method == "t":
            # Use t-distribution (recommended for small samples)
            t_value = stats.t.ppf(1 - alpha/2, n - 1)
            margin_of_error = t_value * std_err
        elif method == "z":
            # Use normal distribution (for large samples)
            z_value = stats.norm.ppf(1 - alpha/2)
            margin_of_error = z_value * std_err
        else:
            raise ValueError(f"Unknown confidence interval method: {method}")
        
        return {
            "mean": mean,
            "standard_error": std_err,
            "confidence_level": confidence_level,
            "margin_of_error": margin_of_error,
            "lower_bound": mean - margin_of_error,
            "upper_bound": mean + margin_of_error,
            "method": method.upper()
        }