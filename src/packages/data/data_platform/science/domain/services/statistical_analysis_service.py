"""Statistical Analysis Service for comprehensive statistical computations and analysis."""

from __future__ import annotations

from typing import Any, Optional, Union
import logging

# TODO: Implement within data platform science domain - from packages.data_science.domain.value_objects.data_distribution import DataDistribution
# TODO: Implement within data platform science domain - from packages.data_science.domain.value_objects.correlation_matrix import CorrelationMatrix


logger = logging.getLogger(__name__)


class StatisticalAnalysisService:
    """Domain service for statistical analysis operations.
    
    This service provides comprehensive statistical analysis capabilities
    including descriptive statistics, distribution fitting, hypothesis testing,
    correlation analysis, and advanced statistical modeling.
    """
    
    def __init__(self) -> None:
        """Initialize the statistical analysis service."""
        self._logger = logger
    
    def analyze_distribution(self, data: Any, 
                           distribution_types: Optional[list[str]] = None) -> list[DataDistribution]:
        """Analyze data distribution and fit multiple distribution types.
        
        Args:
            data: Numerical data to analyze
            distribution_types: List of distribution types to fit
            
        Returns:
            List of fitted distributions ranked by goodness of fit
        """
        if distribution_types is None:
            distribution_types = [
                'normal', 'lognormal', 'exponential', 'gamma', 'beta', 
                'uniform', 'weibull', 'pareto'
            ]
        
        try:
            import numpy as np
            from scipy import stats
            
            # Convert to numpy array
            if not isinstance(data, np.ndarray):
                data = np.array(data)
            
            # Remove invalid values
            data = data[~np.isnan(data)]
            data = data[np.isfinite(data)]
            
            if len(data) < 10:
                raise ValueError("Insufficient data points for distribution analysis")
            
            fitted_distributions = []
            
            for dist_name in distribution_types:
                try:
                    # Get distribution object
                    dist = getattr(stats, dist_name, None)
                    if dist is None:
                        continue
                    
                    # Fit distribution
                    params = dist.fit(data)
                    
                    # Create parameter dictionary
                    param_names = ['loc', 'scale'] if len(params) == 2 else ['shape', 'loc', 'scale']
                    if len(params) > len(param_names):
                        param_names = [f'param_{i}' for i in range(len(params))]
                    
                    param_dict = dict(zip(param_names[:len(params)], params))
                    
                    # Create DataDistribution object
                    distribution = DataDistribution.from_scipy_fit(
                        distribution_name=dist_name,
                        parameters=param_dict,
                        data=data
                    )
                    
                    fitted_distributions.append(distribution)
                    
                except Exception as e:
                    self._logger.warning(f"Failed to fit {dist_name} distribution: {e}")
                    continue
            
            # Sort by goodness of fit
            fitted_distributions.sort(key=lambda d: d.goodness_of_fit, reverse=True)
            
            return fitted_distributions
            
        except ImportError:
            raise ImportError("scipy is required for distribution analysis")
        except Exception as e:
            self._logger.error(f"Distribution analysis failed: {e}")
            raise
    
    def calculate_correlation_matrix(self, data: Any, 
                                   method: str = "pearson",
                                   feature_names: Optional[list[str]] = None) -> CorrelationMatrix:
        """Calculate correlation matrix for multivariate data.
        
        Args:
            data: Multivariate data (2D array or DataFrame)
            method: Correlation method ('pearson', 'spearman', 'kendall')
            feature_names: Names of features/columns
            
        Returns:
            CorrelationMatrix object with correlation analysis
        """
        try:
            import numpy as np
            import pandas as pd
            from scipy.stats import pearsonr, spearmanr, kendalltau
            
            # Convert to DataFrame if needed
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)
            
            if feature_names is not None:
                if len(feature_names) != len(data.columns):
                    raise ValueError("Number of feature names must match number of columns")
                data.columns = feature_names
            else:
                feature_names = [f"feature_{i}" for i in range(len(data.columns))]
                data.columns = feature_names
            
            # Calculate correlation matrix
            if method.lower() == 'pearson':
                corr_matrix = data.corr(method='pearson')
            elif method.lower() == 'spearman':
                corr_matrix = data.corr(method='spearman')
            elif method.lower() == 'kendall':
                corr_matrix = data.corr(method='kendall')
            else:
                raise ValueError(f"Unsupported correlation method: {method}")
            
            # Calculate p-values
            p_values = pd.DataFrame(index=corr_matrix.index, columns=corr_matrix.columns)
            
            for i, col1 in enumerate(corr_matrix.columns):
                for j, col2 in enumerate(corr_matrix.columns):
                    if i <= j:
                        if col1 == col2:
                            p_values.loc[col1, col2] = 0.0
                        else:
                            try:
                                if method.lower() == 'pearson':
                                    _, p_val = pearsonr(data[col1].dropna(), data[col2].dropna())
                                elif method.lower() == 'spearman':
                                    _, p_val = spearmanr(data[col1].dropna(), data[col2].dropna())
                                elif method.lower() == 'kendall':
                                    _, p_val = kendalltau(data[col1].dropna(), data[col2].dropna())
                                
                                p_values.loc[col1, col2] = p_val
                                p_values.loc[col2, col1] = p_val
                            except Exception:
                                p_values.loc[col1, col2] = 1.0
                                p_values.loc[col2, col1] = 1.0
            
            # Create CorrelationMatrix object
            correlation_matrix = CorrelationMatrix.from_pandas_corr(
                correlation_df=corr_matrix,
                method=method,
                p_values_df=p_values,
                sample_size=len(data)
            )
            
            return correlation_matrix
            
        except ImportError:
            raise ImportError("pandas and scipy are required for correlation analysis")
        except Exception as e:
            self._logger.error(f"Correlation analysis failed: {e}")
            raise
    
    def perform_hypothesis_test(self, data1: Any, data2: Optional[Any] = None,
                              test_type: str = "t_test", 
                              alpha: float = 0.05) -> dict[str, Any]:
        """Perform statistical hypothesis tests.
        
        Args:
            data1: First dataset
            data2: Second dataset (for two-sample tests)
            test_type: Type of test to perform
            alpha: Significance level
            
        Returns:
            Test results with statistics and interpretation
        """
        try:
            import numpy as np
            from scipy import stats
            
            # Convert to numpy arrays
            if not isinstance(data1, np.ndarray):
                data1 = np.array(data1)
            
            if data2 is not None and not isinstance(data2, np.ndarray):
                data2 = np.array(data2)
            
            # Remove invalid values
            data1 = data1[~np.isnan(data1)]
            data1 = data1[np.isfinite(data1)]
            
            if data2 is not None:
                data2 = data2[~np.isnan(data2)]
                data2 = data2[np.isfinite(data2)]
            
            result = {
                "test_type": test_type,
                "alpha": alpha,
                "timestamp": self._get_timestamp()
            }
            
            if test_type.lower() == "t_test":
                if data2 is None:
                    # One-sample t-test
                    statistic, p_value = stats.ttest_1samp(data1, 0)
                    result.update({
                        "test_name": "One-sample t-test",
                        "null_hypothesis": "Population mean equals 0",
                        "alternative": "Population mean does not equal 0"
                    })
                else:
                    # Two-sample t-test
                    statistic, p_value = stats.ttest_ind(data1, data2)
                    result.update({
                        "test_name": "Two-sample t-test",
                        "null_hypothesis": "Population means are equal",
                        "alternative": "Population means are not equal"
                    })
            
            elif test_type.lower() == "mann_whitney":
                if data2 is None:
                    raise ValueError("Mann-Whitney test requires two datasets")
                statistic, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                result.update({
                    "test_name": "Mann-Whitney U test",
                    "null_hypothesis": "Distributions are identical",
                    "alternative": "Distributions are different"
                })
            
            elif test_type.lower() == "ks_test":
                if data2 is None:
                    # One-sample KS test against normal distribution
                    statistic, p_value = stats.kstest(data1, 'norm')
                    result.update({
                        "test_name": "One-sample Kolmogorov-Smirnov test",
                        "null_hypothesis": "Data follows normal distribution",
                        "alternative": "Data does not follow normal distribution"
                    })
                else:
                    # Two-sample KS test
                    statistic, p_value = stats.ks_2samp(data1, data2)
                    result.update({
                        "test_name": "Two-sample Kolmogorov-Smirnov test",
                        "null_hypothesis": "Samples come from same distribution",
                        "alternative": "Samples come from different distributions"
                    })
            
            elif test_type.lower() == "shapiro":
                # Normality test
                statistic, p_value = stats.shapiro(data1[:5000])  # Shapiro limited to 5000 samples
                result.update({
                    "test_name": "Shapiro-Wilk normality test",
                    "null_hypothesis": "Data is normally distributed",
                    "alternative": "Data is not normally distributed"
                })
            
            elif test_type.lower() == "anderson":
                # Anderson-Darling normality test
                ad_result = stats.anderson(data1, dist='norm')
                statistic = ad_result.statistic
                critical_values = ad_result.critical_values
                significance_levels = ad_result.significance_level
                
                # Determine p-value approximation
                if statistic < critical_values[0]:
                    p_value = 0.25  # > 15%
                elif statistic < critical_values[1]:
                    p_value = 0.10  # 10-15%
                elif statistic < critical_values[2]:
                    p_value = 0.05  # 5-10%
                elif statistic < critical_values[3]:
                    p_value = 0.025  # 2.5-5%
                elif statistic < critical_values[4]:
                    p_value = 0.01  # 1-2.5%
                else:
                    p_value = 0.005  # < 1%
                
                result.update({
                    "test_name": "Anderson-Darling normality test",
                    "null_hypothesis": "Data is normally distributed",
                    "alternative": "Data is not normally distributed",
                    "critical_values": critical_values.tolist(),
                    "significance_levels": significance_levels.tolist()
                })
            
            else:
                raise ValueError(f"Unsupported test type: {test_type}")
            
            # Add common result fields
            result.update({
                "statistic": float(statistic),
                "p_value": float(p_value),
                "is_significant": p_value < alpha,
                "reject_null": p_value < alpha,
                "sample_size_1": len(data1),
                "sample_size_2": len(data2) if data2 is not None else None
            })
            
            # Add interpretation
            if result["is_significant"]:
                result["interpretation"] = f"Reject null hypothesis at α={alpha} level"
            else:
                result["interpretation"] = f"Fail to reject null hypothesis at α={alpha} level"
            
            return result
            
        except ImportError:
            raise ImportError("scipy is required for hypothesis testing")
        except Exception as e:
            self._logger.error(f"Hypothesis test failed: {e}")
            raise
    
    def calculate_descriptive_statistics(self, data: Any, 
                                       include_advanced: bool = True) -> dict[str, Any]:
        """Calculate comprehensive descriptive statistics.
        
        Args:
            data: Numerical data to analyze
            include_advanced: Whether to include advanced statistics
            
        Returns:
            Dictionary of statistical measures
        """
        try:
            import numpy as np
            from scipy import stats
            
            # Convert to numpy array
            if not isinstance(data, np.ndarray):
                data = np.array(data)
            
            # Remove invalid values
            original_size = len(data)
            data = data[~np.isnan(data)]
            data = data[np.isfinite(data)]
            clean_size = len(data)
            
            if len(data) == 0:
                raise ValueError("No valid data points for analysis")
            
            # Basic statistics
            stats_dict = {
                "count": clean_size,
                "missing_values": original_size - clean_size,
                "mean": float(np.mean(data)),
                "median": float(np.median(data)),
                "mode": float(stats.mode(data)[0]) if len(data) > 0 else None,
                "std": float(np.std(data, ddof=1)),
                "var": float(np.var(data, ddof=1)),
                "min": float(np.min(data)),
                "max": float(np.max(data)),
                "range": float(np.max(data) - np.min(data)),
                "sum": float(np.sum(data))
            }
            
            # Percentiles
            percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
            for p in percentiles:
                stats_dict[f"percentile_{p}"] = float(np.percentile(data, p))
            
            # Quartiles and IQR
            q1, q3 = np.percentile(data, [25, 75])
            stats_dict.update({
                "q1": float(q1),
                "q3": float(q3),
                "iqr": float(q3 - q1)
            })
            
            if include_advanced:
                # Advanced statistics
                stats_dict.update({
                    "skewness": float(stats.skew(data)),
                    "kurtosis": float(stats.kurtosis(data)),
                    "excess_kurtosis": float(stats.kurtosis(data, fisher=True)),
                    "coefficient_of_variation": float(stats_dict["std"] / stats_dict["mean"]) if stats_dict["mean"] != 0 else float('inf'),
                    "mean_absolute_deviation": float(np.mean(np.abs(data - stats_dict["mean"]))),
                    "median_absolute_deviation": float(np.median(np.abs(data - stats_dict["median"]))),
                    "geometric_mean": float(stats.gmean(data[data > 0])) if np.all(data > 0) else None,
                    "harmonic_mean": float(stats.hmean(data[data > 0])) if np.all(data > 0) else None,
                    "trimmed_mean_10": float(stats.trim_mean(data, 0.1)),
                    "trimmed_mean_20": float(stats.trim_mean(data, 0.2))
                })
                
                # Outlier detection using IQR method
                iqr = stats_dict["iqr"]
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = data[(data < lower_bound) | (data > upper_bound)]
                
                stats_dict.update({
                    "outlier_count": len(outliers),
                    "outlier_percentage": float(len(outliers) / len(data) * 100),
                    "outlier_lower_bound": float(lower_bound),
                    "outlier_upper_bound": float(upper_bound),
                    "has_outliers": len(outliers) > 0
                })
                
                # Distribution shape indicators
                if stats_dict["std"] > 0:
                    stats_dict.update({
                        "is_symmetric": abs(stats_dict["skewness"]) < 0.5,
                        "is_normal_like": abs(stats_dict["skewness"]) < 0.5 and abs(stats_dict["excess_kurtosis"]) < 0.5,
                        "distribution_shape": self._classify_distribution_shape(
                            stats_dict["skewness"], 
                            stats_dict["excess_kurtosis"]
                        )
                    })
            
            return stats_dict
            
        except ImportError:
            raise ImportError("scipy is required for advanced statistical calculations")
        except Exception as e:
            self._logger.error(f"Descriptive statistics calculation failed: {e}")
            raise
    
    def perform_outlier_analysis(self, data: Any, 
                                methods: Optional[list[str]] = None) -> dict[str, Any]:
        """Perform comprehensive outlier detection using multiple methods.
        
        Args:
            data: Numerical data to analyze
            methods: List of outlier detection methods to use
            
        Returns:
            Outlier analysis results
        """
        if methods is None:
            methods = ['iqr', 'z_score', 'modified_z_score', 'isolation_forest']
        
        try:
            import numpy as np
            from scipy import stats
            
            # Convert to numpy array
            if not isinstance(data, np.ndarray):
                data = np.array(data)
            
            # Remove invalid values
            data = data[~np.isnan(data)]
            data = data[np.isfinite(data)]
            
            if len(data) < 10:
                raise ValueError("Insufficient data points for outlier analysis")
            
            outlier_results = {
                "total_points": len(data),
                "methods_used": methods,
                "outlier_indices": {},
                "outlier_scores": {},
                "summary": {}
            }
            
            for method in methods:
                try:
                    if method == 'iqr':
                        outliers, scores = self._detect_outliers_iqr(data)
                    elif method == 'z_score':
                        outliers, scores = self._detect_outliers_z_score(data)
                    elif method == 'modified_z_score':
                        outliers, scores = self._detect_outliers_modified_z_score(data)
                    elif method == 'isolation_forest':
                        outliers, scores = self._detect_outliers_isolation_forest(data)
                    else:
                        continue
                    
                    outlier_results["outlier_indices"][method] = outliers.tolist()
                    outlier_results["outlier_scores"][method] = scores.tolist()
                    
                except Exception as e:
                    self._logger.warning(f"Outlier detection method {method} failed: {e}")
                    continue
            
            # Calculate consensus outliers
            if outlier_results["outlier_indices"]:
                consensus_outliers = self._calculate_consensus_outliers(outlier_results["outlier_indices"])
                outlier_results["consensus_outliers"] = consensus_outliers
                
                # Summary statistics
                outlier_results["summary"] = {
                    "consensus_outlier_count": len(consensus_outliers),
                    "consensus_outlier_percentage": len(consensus_outliers) / len(data) * 100,
                    "methods_agreement": self._calculate_methods_agreement(outlier_results["outlier_indices"]),
                    "most_detected_indices": self._get_most_detected_outliers(outlier_results["outlier_indices"])
                }
            
            return outlier_results
            
        except Exception as e:
            self._logger.error(f"Outlier analysis failed: {e}")
            raise
    
    def _detect_outliers_iqr(self, data: Any) -> tuple[Any, Any]:
        """Detect outliers using Interquartile Range method."""
        import numpy as np
        
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outlier_mask = (data < lower_bound) | (data > upper_bound)
        outlier_indices = np.where(outlier_mask)[0]
        
        # Calculate outlier scores (distance from bounds)
        scores = np.zeros(len(data))
        scores[data < lower_bound] = (lower_bound - data[data < lower_bound]) / iqr
        scores[data > upper_bound] = (data[data > upper_bound] - upper_bound) / iqr
        
        return outlier_indices, scores
    
    def _detect_outliers_z_score(self, data: Any, threshold: float = 3.0) -> tuple[Any, Any]:
        """Detect outliers using Z-score method."""
        import numpy as np
        
        z_scores = np.abs((data - np.mean(data)) / np.std(data))
        outlier_indices = np.where(z_scores > threshold)[0]
        
        return outlier_indices, z_scores
    
    def _detect_outliers_modified_z_score(self, data: Any, threshold: float = 3.5) -> tuple[Any, Any]:
        """Detect outliers using Modified Z-score method."""
        import numpy as np
        
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        
        # Handle case where MAD is 0
        if mad == 0:
            mad = np.mean(np.abs(data - median))
        
        if mad == 0:
            return np.array([]), np.zeros(len(data))
        
        modified_z_scores = 0.6745 * (data - median) / mad
        outlier_indices = np.where(np.abs(modified_z_scores) > threshold)[0]
        
        return outlier_indices, np.abs(modified_z_scores)
    
    def _detect_outliers_isolation_forest(self, data: Any) -> tuple[Any, Any]:
        """Detect outliers using Isolation Forest method."""
        try:
            from sklearn.ensemble import IsolationForest
            import numpy as np
            
            # Reshape data for sklearn
            data_reshaped = data.reshape(-1, 1)
            
            # Fit isolation forest
            iso_forest = IsolationForest(contamination='auto', random_state=42)
            outlier_labels = iso_forest.fit_predict(data_reshaped)
            
            # Get outlier scores (negative for outliers)
            outlier_scores = iso_forest.decision_function(data_reshaped)
            
            # Convert to outlier indices
            outlier_indices = np.where(outlier_labels == -1)[0]
            
            # Convert scores to positive values (higher = more outlier-like)
            scores = -outlier_scores
            
            return outlier_indices, scores
            
        except ImportError:
            self._logger.warning("scikit-learn not available, skipping isolation forest method")
            return np.array([]), np.zeros(len(data))
    
    def _calculate_consensus_outliers(self, outlier_indices_dict: dict[str, list[int]]) -> list[int]:
        """Calculate consensus outliers from multiple methods."""
        from collections import Counter
        
        # Count how many methods detected each index as outlier
        all_outliers = []
        for indices in outlier_indices_dict.values():
            all_outliers.extend(indices)
        
        outlier_counts = Counter(all_outliers)
        
        # Consider an outlier if detected by at least half of the methods
        min_detections = max(1, len(outlier_indices_dict) // 2)
        consensus_outliers = [idx for idx, count in outlier_counts.items() 
                            if count >= min_detections]
        
        return sorted(consensus_outliers)
    
    def _calculate_methods_agreement(self, outlier_indices_dict: dict[str, list[int]]) -> float:
        """Calculate agreement between outlier detection methods."""
        if len(outlier_indices_dict) < 2:
            return 1.0
        
        methods = list(outlier_indices_dict.keys())
        agreements = []
        
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                set1 = set(outlier_indices_dict[methods[i]])
                set2 = set(outlier_indices_dict[methods[j]])
                
                # Calculate Jaccard similarity
                intersection = len(set1 & set2)
                union = len(set1 | set2)
                
                similarity = intersection / union if union > 0 else 1.0
                agreements.append(similarity)
        
        return sum(agreements) / len(agreements) if agreements else 1.0
    
    def _get_most_detected_outliers(self, outlier_indices_dict: dict[str, list[int]], 
                                   top_n: int = 10) -> list[tuple[int, int]]:
        """Get the most frequently detected outliers."""
        from collections import Counter
        
        all_outliers = []
        for indices in outlier_indices_dict.values():
            all_outliers.extend(indices)
        
        outlier_counts = Counter(all_outliers)
        
        return outlier_counts.most_common(top_n)
    
    def _classify_distribution_shape(self, skewness: float, excess_kurtosis: float) -> str:
        """Classify distribution shape based on skewness and kurtosis."""
        shape = []
        
        # Skewness classification
        if abs(skewness) < 0.5:
            shape.append("symmetric")
        elif skewness > 0:
            shape.append("right-skewed")
        else:
            shape.append("left-skewed")
        
        # Kurtosis classification
        if abs(excess_kurtosis) < 0.5:
            shape.append("mesokurtic")
        elif excess_kurtosis > 0:
            shape.append("leptokurtic")
        else:
            shape.append("platykurtic")
        
        return ", ".join(shape)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.utcnow().isoformat()