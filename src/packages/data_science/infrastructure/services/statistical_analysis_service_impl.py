"""Implementation of statistical analysis service using pandas and scipy."""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kendalltau, mannwhitneyu, chi2_contingency
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
from datetime import datetime
from uuid import uuid4

from ...domain.services.statistical_analysis_service import IStatisticalAnalysisService
from ...domain.entities.analysis_job import AnalysisJob
from ...domain.entities.statistical_profile import StatisticalProfile
from ...domain.value_objects.statistical_metrics import StatisticalMetrics
from ...domain.value_objects.correlation_matrix import CorrelationMatrix, CorrelationType
from ...domain.value_objects.data_distribution import DataDistribution, DistributionType, DistributionTest


class StatisticalAnalysisServiceImpl(IStatisticalAnalysisService):
    """Implementation of statistical analysis service using scientific Python libraries."""
    
    async def perform_descriptive_analysis(self, dataset: Any, 
                                         analysis_config: Dict[str, Any]) -> StatisticalProfile:
        """Perform comprehensive descriptive statistical analysis."""
        if isinstance(dataset, pd.DataFrame):
            df = dataset
        else:
            # Assume dataset is a path or other format - would implement proper loading
            raise NotImplementedError("Only pandas DataFrame currently supported")
        
        # Calculate descriptive statistics
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        descriptive_stats = {}
        
        for col in numeric_columns:
            series = df[col].dropna()
            descriptive_stats[col] = {
                "count": len(series),
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "max": float(series.max()),
                "median": float(series.median()),
                "q25": float(series.quantile(0.25)),
                "q75": float(series.quantile(0.75)),
                "skewness": float(series.skew()),
                "kurtosis": float(series.kurtosis())
            }
        
        # Create statistical profile (mock implementation)
        # In real implementation, this would create proper StatisticalProfile entity
        return {
            "descriptive_statistics": descriptive_stats,
            "dataset_summary": {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "numeric_columns": len(numeric_columns),
                "missing_values": df.isnull().sum().to_dict()
            }
        }
    
    async def perform_correlation_analysis(self, dataset: Any,
                                         features: Optional[List[str]] = None,
                                         method: str = "pearson") -> CorrelationMatrix:
        """Perform correlation analysis between features."""
        if isinstance(dataset, pd.DataFrame):
            df = dataset
        else:
            raise NotImplementedError("Only pandas DataFrame currently supported")
        
        if features:
            df = df[features]
        else:
            df = df.select_dtypes(include=[np.number])
        
        # Calculate correlation matrix
        if method == "pearson":
            corr_matrix = df.corr(method='pearson')
            corr_type = CorrelationType.PEARSON
        elif method == "spearman":
            corr_matrix = df.corr(method='spearman')
            corr_type = CorrelationType.SPEARMAN
        elif method == "kendall":
            corr_matrix = df.corr(method='kendall')
            corr_type = CorrelationType.KENDALL
        else:
            raise ValueError(f"Unsupported correlation method: {method}")
        
        # Calculate p-values
        n = len(df)
        p_values = np.zeros_like(corr_matrix.values)
        
        for i, col1 in enumerate(corr_matrix.columns):
            for j, col2 in enumerate(corr_matrix.columns):
                if i != j:
                    if method == "pearson":
                        _, p_val = pearsonr(df[col1].dropna(), df[col2].dropna())
                    elif method == "spearman":
                        _, p_val = spearmanr(df[col1].dropna(), df[col2].dropna())
                    elif method == "kendall":
                        _, p_val = kendalltau(df[col1].dropna(), df[col2].dropna())
                    p_values[i, j] = p_val
        
        # Calculate eigenvalues for multicollinearity analysis
        eigenvals = np.linalg.eigvals(corr_matrix.values)
        condition_index = np.sqrt(np.max(eigenvals) / np.min(eigenvals[eigenvals > 1e-10]))
        
        return CorrelationMatrix(
            features=list(corr_matrix.columns),
            correlation_matrix=corr_matrix.values.tolist(),
            correlation_type=corr_type,
            p_value_matrix=p_values.tolist(),
            significance_level=0.05,
            eigenvalues=eigenvals.tolist(),
            determinant=float(np.linalg.det(corr_matrix.values)),
            condition_index=float(condition_index),
            average_correlation=float(np.mean(np.abs(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)])))
        )
    
    async def analyze_data_distribution(self, dataset: Any, feature: str) -> DataDistribution:
        """Analyze the distribution of a specific feature."""
        if isinstance(dataset, pd.DataFrame):
            df = dataset
        else:
            raise NotImplementedError("Only pandas DataFrame currently supported")
        
        if feature not in df.columns:
            raise ValueError(f"Feature '{feature}' not found in dataset")
        
        series = df[feature].dropna()
        
        # Fit normal distribution
        mu, sigma = stats.norm.fit(series)
        
        # Perform normality tests
        shapiro_stat, shapiro_p = stats.shapiro(series.sample(min(5000, len(series))))
        ks_stat, ks_p = stats.kstest(series, 'norm', args=(mu, sigma))
        
        # Create distribution tests
        normality_tests = [
            DistributionTest(
                test_name="Shapiro-Wilk",
                statistic=float(shapiro_stat),
                p_value=float(shapiro_p),
                critical_value=None,
                confidence_level=0.95,
                null_hypothesis="Data is normally distributed",
                interpretation="Normal" if shapiro_p > 0.05 else "Not normal"
            ),
            DistributionTest(
                test_name="Kolmogorov-Smirnov",
                statistic=float(ks_stat),
                p_value=float(ks_p),
                critical_value=None,
                confidence_level=0.95,
                null_hypothesis="Data follows normal distribution",
                interpretation="Normal" if ks_p > 0.05 else "Not normal"
            )
        ]
        
        return DataDistribution(
            feature_name=feature,
            distribution_type=DistributionType.NORMAL,
            sample_size=len(series),
            parameters={"mu": mu, "sigma": sigma},
            descriptive_stats={
                "mean": float(series.mean()),
                "std": float(series.std()),
                "skewness": float(series.skew()),
                "kurtosis": float(series.kurtosis()),
                "min": float(series.min()),
                "max": float(series.max())
            },
            normality_tests=normality_tests,
            goodness_of_fit_score=float(max(shapiro_p, ks_p)),
            is_normal=(shapiro_p > 0.05 and ks_p > 0.05)
        )
    
    async def detect_outliers(self, dataset: Any, 
                            features: Optional[List[str]] = None,
                            methods: Optional[List[str]] = None) -> Dict[str, Any]:
        """Detect outliers in the dataset using multiple methods."""
        if isinstance(dataset, pd.DataFrame):
            df = dataset
        else:
            raise NotImplementedError("Only pandas DataFrame currently supported")
        
        if features is None:
            features = list(df.select_dtypes(include=[np.number]).columns)
        
        methods = methods or ["iqr", "zscore", "isolation_forest"]
        outlier_results = {}
        
        for feature in features:
            if feature not in df.columns:
                continue
                
            series = df[feature].dropna()
            feature_outliers = {}
            
            # IQR method
            if "iqr" in methods:
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                iqr_outliers = series[(series < lower_bound) | (series > upper_bound)]
                
                feature_outliers["iqr"] = {
                    "outlier_count": len(iqr_outliers),
                    "outlier_percentage": (len(iqr_outliers) / len(series)) * 100,
                    "lower_bound": float(lower_bound),
                    "upper_bound": float(upper_bound),
                    "outlier_indices": iqr_outliers.index.tolist()
                }
            
            # Z-score method
            if "zscore" in methods:
                z_scores = np.abs(stats.zscore(series))
                zscore_outliers = series[z_scores > 3]
                
                feature_outliers["zscore"] = {
                    "outlier_count": len(zscore_outliers),
                    "outlier_percentage": (len(zscore_outliers) / len(series)) * 100,
                    "threshold": 3.0,
                    "outlier_indices": zscore_outliers.index.tolist()
                }
            
            # Isolation Forest method
            if "isolation_forest" in methods and len(series) >= 10:
                try:
                    isolation_forest = IsolationForest(contamination=0.1, random_state=42)
                    outlier_labels = isolation_forest.fit_predict(series.values.reshape(-1, 1))
                    isolation_outliers = series[outlier_labels == -1]
                    
                    feature_outliers["isolation_forest"] = {
                        "outlier_count": len(isolation_outliers),
                        "outlier_percentage": (len(isolation_outliers) / len(series)) * 100,
                        "contamination": 0.1,
                        "outlier_indices": isolation_outliers.index.tolist()
                    }
                except Exception as e:
                    feature_outliers["isolation_forest"] = {
                        "error": f"Isolation Forest failed: {str(e)}"
                    }
            
            # Modified Z-score (MAD-based)
            if "modified_zscore" in methods:
                median = series.median()
                mad = np.median(np.abs(series - median))
                modified_z_scores = 0.6745 * (series - median) / mad if mad != 0 else np.zeros_like(series)
                mad_outliers = series[np.abs(modified_z_scores) > 3.5]
                
                feature_outliers["modified_zscore"] = {
                    "outlier_count": len(mad_outliers),
                    "outlier_percentage": (len(mad_outliers) / len(series)) * 100,
                    "threshold": 3.5,
                    "outlier_indices": mad_outliers.index.tolist(),
                    "median_absolute_deviation": float(mad)
                }
            
            outlier_results[feature] = feature_outliers
        
        return outlier_results
    
    async def perform_hypothesis_test(self, dataset: Any,
                                    test_type: str,
                                    test_config: Dict[str, Any]) -> Dict[str, Any]:
        """Perform statistical hypothesis testing."""
        if isinstance(dataset, pd.DataFrame):
            df = dataset
        else:
            raise NotImplementedError("Only pandas DataFrame currently supported")
        
        if test_type == "t_test_one_sample":
            sample = df[test_config["column"]].dropna()
            null_mean = test_config.get("null_mean", 0)
            statistic, p_value = stats.ttest_1samp(sample, null_mean)
            
            return {
                "test_type": "One-sample t-test",
                "statistic": float(statistic),
                "p_value": float(p_value),
                "null_hypothesis": f"Mean equals {null_mean}",
                "interpretation": "Reject null hypothesis" if p_value < 0.05 else "Fail to reject null hypothesis"
            }
        
        elif test_type == "t_test_independent":
            group1 = df[df[test_config["group_column"]] == test_config["group1"]][test_config["value_column"]].dropna()
            group2 = df[df[test_config["group_column"]] == test_config["group2"]][test_config["value_column"]].dropna()
            statistic, p_value = stats.ttest_ind(group1, group2)
            
            return {
                "test_type": "Independent samples t-test",
                "statistic": float(statistic),
                "p_value": float(p_value),
                "null_hypothesis": "Group means are equal",
                "interpretation": "Reject null hypothesis" if p_value < 0.05 else "Fail to reject null hypothesis",
                "effect_size": float(abs(group1.mean() - group2.mean()) / np.sqrt((group1.var() + group2.var()) / 2))
            }
        
        elif test_type == "t_test_paired":
            before = df[test_config["before_column"]].dropna()
            after = df[test_config["after_column"]].dropna()
            # Ensure paired data
            common_indices = before.index.intersection(after.index)
            before_paired = before.loc[common_indices]
            after_paired = after.loc[common_indices]
            
            statistic, p_value = stats.ttest_rel(before_paired, after_paired)
            
            return {
                "test_type": "Paired samples t-test",
                "statistic": float(statistic),
                "p_value": float(p_value),
                "null_hypothesis": "Mean difference equals zero",
                "interpretation": "Reject null hypothesis" if p_value < 0.05 else "Fail to reject null hypothesis",
                "mean_difference": float((after_paired - before_paired).mean()),
                "pairs_count": len(common_indices)
            }
        
        elif test_type == "chi_square_goodness_of_fit":
            observed = df[test_config["category_column"]].value_counts().sort_index()
            expected = test_config.get("expected_frequencies", None)
            
            if expected is None:
                # Assume uniform distribution
                expected = [len(df) / len(observed)] * len(observed)
            
            statistic, p_value = stats.chisquare(observed, expected)
            
            return {
                "test_type": "Chi-square goodness of fit test",
                "statistic": float(statistic),
                "p_value": float(p_value),
                "degrees_of_freedom": len(observed) - 1,
                "null_hypothesis": "Data follows expected distribution",
                "interpretation": "Reject null hypothesis" if p_value < 0.05 else "Fail to reject null hypothesis",
                "observed_frequencies": observed.tolist(),
                "expected_frequencies": expected
            }
        
        elif test_type == "chi_square_independence":
            contingency_table = pd.crosstab(df[test_config["variable1"]], df[test_config["variable2"]])
            statistic, p_value, dof, expected = chi2_contingency(contingency_table)
            
            # Calculate Cramér's V
            n = contingency_table.sum().sum()
            cramers_v = np.sqrt(statistic / (n * (min(contingency_table.shape) - 1)))
            
            return {
                "test_type": "Chi-square test of independence",
                "statistic": float(statistic),
                "p_value": float(p_value),
                "degrees_of_freedom": int(dof),
                "null_hypothesis": "Variables are independent",
                "interpretation": "Reject null hypothesis" if p_value < 0.05 else "Fail to reject null hypothesis",
                "cramers_v": float(cramers_v),
                "contingency_table": contingency_table.to_dict()
            }
        
        elif test_type == "anova_one_way":
            groups = []
            group_names = test_config["groups"]
            value_column = test_config["value_column"]
            group_column = test_config["group_column"]
            
            for group_name in group_names:
                group_data = df[df[group_column] == group_name][value_column].dropna()
                groups.append(group_data)
            
            statistic, p_value = stats.f_oneway(*groups)
            
            # Calculate effect size (eta-squared)
            group_means = [group.mean() for group in groups]
            overall_mean = np.concatenate(groups).mean()
            ss_between = sum(len(group) * (group_mean - overall_mean)**2 for group, group_mean in zip(groups, group_means))
            ss_total = sum((np.concatenate(groups) - overall_mean)**2)
            eta_squared = ss_between / ss_total if ss_total != 0 else 0
            
            return {
                "test_type": "One-way ANOVA",
                "statistic": float(statistic),
                "p_value": float(p_value),
                "degrees_of_freedom_between": len(groups) - 1,
                "degrees_of_freedom_within": len(np.concatenate(groups)) - len(groups),
                "null_hypothesis": "All group means are equal",
                "interpretation": "Reject null hypothesis" if p_value < 0.05 else "Fail to reject null hypothesis",
                "eta_squared": float(eta_squared),
                "group_means": [float(mean) for mean in group_means],
                "group_sizes": [len(group) for group in groups]
            }
        
        elif test_type == "mann_whitney_u":
            group1 = df[df[test_config["group_column"]] == test_config["group1"]][test_config["value_column"]].dropna()
            group2 = df[df[test_config["group_column"]] == test_config["group2"]][test_config["value_column"]].dropna()
            
            statistic, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
            
            return {
                "test_type": "Mann-Whitney U test",
                "statistic": float(statistic),
                "p_value": float(p_value),
                "null_hypothesis": "Distributions are equal",
                "interpretation": "Reject null hypothesis" if p_value < 0.05 else "Fail to reject null hypothesis",
                "group1_median": float(group1.median()),
                "group2_median": float(group2.median())
            }
        
        elif test_type == "wilcoxon_signed_rank":
            before = df[test_config["before_column"]].dropna()
            after = df[test_config["after_column"]].dropna()
            # Ensure paired data
            common_indices = before.index.intersection(after.index)
            before_paired = before.loc[common_indices]
            after_paired = after.loc[common_indices]
            
            statistic, p_value = stats.wilcoxon(before_paired, after_paired)
            
            return {
                "test_type": "Wilcoxon signed-rank test",
                "statistic": float(statistic),
                "p_value": float(p_value),
                "null_hypothesis": "Median difference equals zero",
                "interpretation": "Reject null hypothesis" if p_value < 0.05 else "Fail to reject null hypothesis",
                "median_difference": float((after_paired - before_paired).median()),
                "pairs_count": len(common_indices)
            }
        
        elif test_type == "kruskal_wallis":
            groups = []
            group_names = test_config["groups"]
            value_column = test_config["value_column"]
            group_column = test_config["group_column"]
            
            for group_name in group_names:
                group_data = df[df[group_column] == group_name][value_column].dropna()
                groups.append(group_data)
            
            statistic, p_value = stats.kruskal(*groups)
            
            return {
                "test_type": "Kruskal-Wallis test",
                "statistic": float(statistic),
                "p_value": float(p_value),
                "degrees_of_freedom": len(groups) - 1,
                "null_hypothesis": "All group distributions are equal",
                "interpretation": "Reject null hypothesis" if p_value < 0.05 else "Fail to reject null hypothesis",
                "group_medians": [float(group.median()) for group in groups],
                "group_sizes": [len(group) for group in groups]
            }
        
        else:
            raise ValueError(f"Unsupported test type: {test_type}. Supported tests: t_test_one_sample, t_test_independent, t_test_paired, chi_square_goodness_of_fit, chi_square_independence, anova_one_way, mann_whitney_u, wilcoxon_signed_rank, kruskal_wallis")
    
    async def calculate_feature_statistics(self, dataset: Any, feature: str) -> StatisticalMetrics:
        """Calculate comprehensive statistics for a single feature."""
        if isinstance(dataset, pd.DataFrame):
            df = dataset
        else:
            raise NotImplementedError("Only pandas DataFrame currently supported")
        
        if feature not in df.columns:
            raise ValueError(f"Feature '{feature}' not found in dataset")
        
        series = df[feature].dropna()
        
        # Handle different data types
        if pd.api.types.is_numeric_dtype(series):
            # Numeric statistics
            stats_dict = {
                "count": len(series),
                "mean": float(series.mean()),
                "median": float(series.median()),
                "mode": series.mode().iloc[0] if not series.mode().empty else None,
                "std": float(series.std()),
                "variance": float(series.var()),
                "min": float(series.min()),
                "max": float(series.max()),
                "range": float(series.max() - series.min()),
                "q25": float(series.quantile(0.25)),
                "q75": float(series.quantile(0.75)),
                "iqr": float(series.quantile(0.75) - series.quantile(0.25)),
                "skewness": float(series.skew()),
                "kurtosis": float(series.kurtosis()),
                "coefficient_of_variation": float(series.std() / series.mean()) if series.mean() != 0 else None
            }
            
            # Additional percentiles
            percentiles = [1, 5, 10, 90, 95, 99]
            for p in percentiles:
                stats_dict[f"p{p}"] = float(series.quantile(p/100))
            
            # Outlier detection
            iqr = stats_dict["iqr"]
            q1, q3 = stats_dict["q25"], stats_dict["q75"]
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = series[(series < lower_bound) | (series > upper_bound)]
            
            stats_dict.update({
                "outlier_count": len(outliers),
                "outlier_percentage": (len(outliers) / len(series)) * 100,
                "outlier_lower_bound": float(lower_bound),
                "outlier_upper_bound": float(upper_bound)
            })
            
        else:
            # Categorical statistics
            value_counts = series.value_counts()
            stats_dict = {
                "count": len(series),
                "unique_count": series.nunique(),
                "most_frequent": value_counts.index[0] if not value_counts.empty else None,
                "most_frequent_count": int(value_counts.iloc[0]) if not value_counts.empty else 0,
                "least_frequent": value_counts.index[-1] if not value_counts.empty else None,
                "least_frequent_count": int(value_counts.iloc[-1]) if not value_counts.empty else 0,
                "entropy": float(-sum((p := value_counts / len(series)) * np.log2(p + 1e-10))),
                "concentration_ratio": float(value_counts.iloc[0] / len(series)) if not value_counts.empty else 0
            }
        
        return StatisticalMetrics(
            feature_name=feature,
            data_type=str(series.dtype),
            sample_size=len(series),
            missing_count=df[feature].isnull().sum(),
            missing_percentage=(df[feature].isnull().sum() / len(df)) * 100,
            mean=stats_dict.get("mean"),
            median=stats_dict.get("median"),
            mode=stats_dict.get("mode"),
            standard_deviation=stats_dict.get("std"),
            variance=stats_dict.get("variance"),
            minimum=stats_dict.get("min"),
            maximum=stats_dict.get("max"),
            range_value=stats_dict.get("range"),
            quantile_25=stats_dict.get("q25"),
            quantile_75=stats_dict.get("q75"),
            interquartile_range=stats_dict.get("iqr"),
            skewness=stats_dict.get("skewness"),
            kurtosis=stats_dict.get("kurtosis"),
            coefficient_of_variation=stats_dict.get("coefficient_of_variation"),
            unique_count=stats_dict.get("unique_count", series.nunique()),
            outlier_count=stats_dict.get("outlier_count", 0),
            entropy=stats_dict.get("entropy"),
            additional_metrics=stats_dict
        )
    
    async def compare_distributions(self, dataset1: Any, dataset2: Any, feature: str) -> Dict[str, Any]:
        """Compare distributions between two datasets for a feature."""
        # Convert datasets to DataFrames
        if isinstance(dataset1, pd.DataFrame):
            df1 = dataset1
        else:
            raise NotImplementedError("Only pandas DataFrame currently supported")
            
        if isinstance(dataset2, pd.DataFrame):
            df2 = dataset2
        else:
            raise NotImplementedError("Only pandas DataFrame currently supported")
        
        if feature not in df1.columns or feature not in df2.columns:
            raise ValueError(f"Feature '{feature}' not found in one or both datasets")
        
        series1 = df1[feature].dropna()
        series2 = df2[feature].dropna()
        
        comparison_results = {
            "feature": feature,
            "dataset1_size": len(series1),
            "dataset2_size": len(series2),
            "tests": {},
            "effect_sizes": {},
            "descriptive_comparison": {}
        }
        
        if pd.api.types.is_numeric_dtype(series1) and pd.api.types.is_numeric_dtype(series2):
            # Numeric comparison
            
            # Descriptive statistics comparison
            comparison_results["descriptive_comparison"] = {
                "mean_diff": float(series1.mean() - series2.mean()),
                "median_diff": float(series1.median() - series2.median()),
                "std_ratio": float(series1.std() / series2.std()) if series2.std() != 0 else None,
                "variance_ratio": float(series1.var() / series2.var()) if series2.var() != 0 else None
            }
            
            # Statistical tests
            # Shapiro-Wilk normality test (for small samples)
            if len(series1) <= 5000 and len(series2) <= 5000:
                _, sw1_p = stats.shapiro(series1.sample(min(5000, len(series1))))
                _, sw2_p = stats.shapiro(series2.sample(min(5000, len(series2))))
                both_normal = sw1_p > 0.05 and sw2_p > 0.05
            else:
                both_normal = False
            
            # Choose appropriate test based on normality
            if both_normal and len(series1) > 30 and len(series2) > 30:
                # Parametric test: Independent t-test
                statistic, p_value = stats.ttest_ind(series1, series2)
                test_name = "Independent t-test"
                test_type = "parametric"
            else:
                # Non-parametric test: Mann-Whitney U test
                statistic, p_value = mannwhitneyu(series1, series2, alternative='two-sided')
                test_name = "Mann-Whitney U test"
                test_type = "non-parametric"
            
            comparison_results["tests"]["location_test"] = {
                "test_name": test_name,
                "test_type": test_type,
                "statistic": float(statistic),
                "p_value": float(p_value),
                "significant": p_value < 0.05,
                "interpretation": "Distributions differ significantly" if p_value < 0.05 else "No significant difference"
            }
            
            # Kolmogorov-Smirnov test for distribution shape
            ks_stat, ks_p = stats.ks_2samp(series1, series2)
            comparison_results["tests"]["distribution_test"] = {
                "test_name": "Kolmogorov-Smirnov",
                "statistic": float(ks_stat),
                "p_value": float(ks_p),
                "significant": ks_p < 0.05,
                "interpretation": "Distributions differ significantly" if ks_p < 0.05 else "Distributions are similar"
            }
            
            # Effect sizes
            # Cohen's d for effect size
            pooled_std = np.sqrt(((len(series1) - 1) * series1.var() + (len(series2) - 1) * series2.var()) / 
                               (len(series1) + len(series2) - 2))
            cohens_d = (series1.mean() - series2.mean()) / pooled_std if pooled_std != 0 else 0
            
            comparison_results["effect_sizes"]["cohens_d"] = {
                "value": float(cohens_d),
                "magnitude": self._interpret_cohens_d(cohens_d),
                "description": "Standardized difference in means"
            }
            
        else:
            # Categorical comparison
            # Chi-square test of independence
            contingency_table = pd.crosstab(
                pd.concat([series1, series2]), 
                pd.concat([pd.Series(['Dataset1'] * len(series1)), pd.Series(['Dataset2'] * len(series2))])
            )
            
            chi2_stat, chi2_p, dof, expected = chi2_contingency(contingency_table)
            
            comparison_results["tests"]["independence_test"] = {
                "test_name": "Chi-square test of independence",
                "statistic": float(chi2_stat),
                "p_value": float(chi2_p),
                "degrees_of_freedom": int(dof),
                "significant": chi2_p < 0.05,
                "interpretation": "Distributions differ significantly" if chi2_p < 0.05 else "No significant difference"
            }
            
            # Cramér's V for effect size
            n = contingency_table.sum().sum()
            cramers_v = np.sqrt(chi2_stat / (n * (min(contingency_table.shape) - 1)))
            
            comparison_results["effect_sizes"]["cramers_v"] = {
                "value": float(cramers_v),
                "magnitude": self._interpret_cramers_v(cramers_v),
                "description": "Association strength between categorical variables"
            }
        
        return comparison_results
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _interpret_cramers_v(self, v: float) -> str:
        """Interpret Cramér's V effect size."""
        if v < 0.1:
            return "negligible"
        elif v < 0.3:
            return "small"
        elif v < 0.5:
            return "medium"
        else:
            return "large"
    
    async def analyze_time_series(self, dataset: Any, timestamp_column: str, 
                                value_column: str, analysis_config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze time series data for trends, seasonality, and patterns."""
        if isinstance(dataset, pd.DataFrame):
            df = dataset.copy()
        else:
            raise NotImplementedError("Only pandas DataFrame currently supported")
        
        if timestamp_column not in df.columns or value_column not in df.columns:
            raise ValueError(f"Columns '{timestamp_column}' or '{value_column}' not found in dataset")
        
        # Convert timestamp column to datetime
        df[timestamp_column] = pd.to_datetime(df[timestamp_column])
        df = df.sort_values(timestamp_column).dropna(subset=[value_column])
        
        series = df.set_index(timestamp_column)[value_column]
        
        # Basic time series characteristics
        results = {
            "time_range": {
                "start": series.index.min().isoformat(),
                "end": series.index.max().isoformat(),
                "duration_days": (series.index.max() - series.index.min()).days,
                "observations": len(series)
            },
            "descriptive_stats": {
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "max": float(series.max()),
                "trend_direction": self._determine_trend_direction(series)
            }
        }
        
        # Stationarity tests
        try:
            from statsmodels.tsa.stattools import adfuller
            adf_result = adfuller(series.dropna())
            results["stationarity"] = {
                "adf_statistic": float(adf_result[0]),
                "adf_p_value": float(adf_result[1]),
                "is_stationary": adf_result[1] < 0.05,
                "critical_values": {str(k): float(v) for k, v in adf_result[4].items()}
            }
        except ImportError:
            results["stationarity"] = {"error": "statsmodels not available for advanced time series analysis"}
        
        # Autocorrelation analysis
        autocorr_lags = min(40, len(series) // 4)
        autocorrelations = [series.autocorr(lag=i) for i in range(1, autocorr_lags + 1)]
        results["autocorrelation"] = {
            "lag_1": float(autocorrelations[0]) if autocorrelations else None,
            "max_autocorr": float(max(autocorrelations)) if autocorrelations else None,
            "max_autocorr_lag": int(autocorrelations.index(max(autocorrelations)) + 1) if autocorrelations else None
        }
        
        # Simple trend analysis
        if len(series) > 2:
            x = np.arange(len(series))
            y = series.values
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            results["trend_analysis"] = {
                "linear_slope": float(slope),
                "r_squared": float(r_value ** 2),
                "trend_p_value": float(p_value),
                "significant_trend": p_value < 0.05,
                "trend_interpretation": self._interpret_trend(slope, p_value)
            }
        
        # Seasonality detection (basic)
        if analysis_config.get("detect_seasonality", True) and len(series) >= 24:
            # Simple seasonality test using autocorrelation
            seasonal_lags = [7, 30, 365] if 'freq' not in analysis_config else analysis_config['seasonal_lags']
            seasonality_results = {}
            
            for lag in seasonal_lags:
                if lag < len(series):
                    seasonal_autocorr = series.autocorr(lag=lag)
                    seasonality_results[f"lag_{lag}"] = {
                        "autocorrelation": float(seasonal_autocorr),
                        "likely_seasonal": abs(seasonal_autocorr) > 0.3
                    }
            
            results["seasonality"] = seasonality_results
        
        return results
    
    def _determine_trend_direction(self, series: pd.Series) -> str:
        """Determine the overall trend direction."""
        if len(series) < 2:
            return "insufficient_data"
        
        first_half_mean = series.iloc[:len(series)//2].mean()
        second_half_mean = series.iloc[len(series)//2:].mean()
        
        if second_half_mean > first_half_mean * 1.05:
            return "increasing"
        elif second_half_mean < first_half_mean * 0.95:
            return "decreasing"
        else:
            return "stable"
    
    def _interpret_trend(self, slope: float, p_value: float) -> str:
        """Interpret trend analysis results."""
        if p_value >= 0.05:
            return "no_significant_trend"
        elif slope > 0:
            return "significant_upward_trend"
        else:
            return "significant_downward_trend"
    
    async def validate_analysis_requirements(self, dataset: Any, analysis_type: str) -> Dict[str, Any]:
        """Validate that dataset meets requirements for analysis type."""
        if isinstance(dataset, pd.DataFrame):
            df = dataset
        else:
            raise NotImplementedError("Only pandas DataFrame currently supported")
        
        validation_results = {
            "analysis_type": analysis_type,
            "dataset_valid": True,
            "warnings": [],
            "errors": [],
            "requirements_met": True,
            "dataset_info": {
                "shape": df.shape,
                "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
                "missing_values_total": df.isnull().sum().sum(),
                "missing_percentage": (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
            }
        }
        
        # General validations
        if df.empty:
            validation_results["errors"].append("Dataset is empty")
            validation_results["dataset_valid"] = False
        
        if df.shape[0] < 2:
            validation_results["errors"].append("Dataset must have at least 2 rows")
            validation_results["dataset_valid"] = False
        
        # Analysis-specific validations
        if analysis_type == "descriptive_statistics":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                validation_results["warnings"].append("No numeric columns found for descriptive statistics")
        
        elif analysis_type == "correlation_analysis":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 2:
                validation_results["errors"].append("Correlation analysis requires at least 2 numeric columns")
                validation_results["requirements_met"] = False
            
            # Check for sufficient variance
            for col in numeric_cols:
                if df[col].std() == 0:
                    validation_results["warnings"].append(f"Column '{col}' has zero variance")
        
        elif analysis_type == "hypothesis_testing":
            if df.shape[0] < 30:
                validation_results["warnings"].append("Sample size < 30 may affect test validity")
        
        elif analysis_type == "time_series":
            # Requires timestamp validation
            datetime_cols = df.select_dtypes(include=['datetime64']).columns
            if len(datetime_cols) == 0:
                validation_results["errors"].append("Time series analysis requires a datetime column")
                validation_results["requirements_met"] = False
        
        elif analysis_type == "outlier_detection":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                validation_results["errors"].append("Outlier detection requires numeric columns")
                validation_results["requirements_met"] = False
        
        # Data quality checks
        missing_percentage = validation_results["dataset_info"]["missing_percentage"]
        if missing_percentage > 50:
            validation_results["errors"].append(f"Dataset has {missing_percentage:.1f}% missing values")
            validation_results["requirements_met"] = False
        elif missing_percentage > 20:
            validation_results["warnings"].append(f"Dataset has {missing_percentage:.1f}% missing values")
        
        # Memory usage warning
        if validation_results["dataset_info"]["memory_usage_mb"] > 1000:
            validation_results["warnings"].append("Large dataset may require additional memory")
        
        validation_results["dataset_valid"] = validation_results["requirements_met"] and len(validation_results["errors"]) == 0
        
        return validation_results
    
    async def generate_analysis_report(self, analysis_job: AnalysisJob, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive analysis report."""
        report = {
            "report_id": str(uuid4()),
            "generated_at": datetime.utcnow().isoformat(),
            "analysis_job": {
                "job_id": str(analysis_job.id),
                "job_name": analysis_job.job_name,
                "analysis_type": analysis_job.analysis_type,
                "status": analysis_job.status.value,
                "created_at": analysis_job.created_at.isoformat(),
                "completed_at": analysis_job.completed_at.isoformat() if analysis_job.completed_at else None
            },
            "executive_summary": {},
            "detailed_results": results,
            "recommendations": [],
            "data_quality_assessment": {},
            "statistical_significance": {},
            "visualizations": [],
            "export_formats": ["json", "csv", "pdf", "html"]
        }
        
        # Generate executive summary based on analysis type
        if analysis_job.analysis_type == "descriptive_statistics":
            report["executive_summary"] = self._generate_descriptive_summary(results)
        elif analysis_job.analysis_type == "correlation_analysis":
            report["executive_summary"] = self._generate_correlation_summary(results)
        elif analysis_job.analysis_type == "hypothesis_testing":
            report["executive_summary"] = self._generate_hypothesis_summary(results)
        
        # Generate recommendations
        report["recommendations"] = self._generate_recommendations(analysis_job.analysis_type, results)
        
        # Statistical significance assessment
        report["statistical_significance"] = self._assess_statistical_significance(results)
        
        return report
    
    def _generate_descriptive_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary for descriptive analysis."""
        return {
            "summary_type": "descriptive_statistics",
            "key_findings": [
                "Descriptive statistics calculated for all numeric variables",
                f"Dataset contains {results.get('dataset_summary', {}).get('total_rows', 0)} observations",
                f"Analysis included {results.get('dataset_summary', {}).get('numeric_columns', 0)} numeric variables"
            ],
            "data_overview": results.get('dataset_summary', {}),
            "quality_score": self._calculate_data_quality_score(results)
        }
    
    def _generate_correlation_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary for correlation analysis."""
        # Extract key correlation insights
        if hasattr(results, 'correlation_matrix'):
            corr_matrix = np.array(results.correlation_matrix)
            high_corr_count = np.sum(np.abs(corr_matrix) > 0.7) - len(corr_matrix)  # Exclude diagonal
            avg_corr = results.average_correlation
        else:
            high_corr_count = 0
            avg_corr = 0
        
        return {
            "summary_type": "correlation_analysis",
            "key_findings": [
                f"Found {high_corr_count} strong correlations (|r| > 0.7)",
                f"Average correlation strength: {avg_corr:.3f}",
                "Multicollinearity assessment completed"
            ],
            "correlation_strength": "high" if avg_corr > 0.5 else "moderate" if avg_corr > 0.3 else "low",
            "multicollinearity_risk": "high" if high_corr_count > 0 else "low"
        }
    
    def _generate_hypothesis_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary for hypothesis testing."""
        return {
            "summary_type": "hypothesis_testing",
            "key_findings": [
                f"Test performed: {results.get('test_type', 'Unknown')}",
                f"P-value: {results.get('p_value', 'N/A')}",
                f"Result: {results.get('interpretation', 'N/A')}"
            ],
            "statistical_significance": results.get('p_value', 1) < 0.05,
            "effect_size": "moderate"  # This would be calculated based on specific test
        }
    
    def _generate_recommendations(self, analysis_type: str, results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on analysis results."""
        recommendations = []
        
        if analysis_type == "descriptive_statistics":
            recommendations.extend([
                "Review variables with high missing percentages for data collection improvements",
                "Investigate outliers identified in the analysis",
                "Consider data transformation for highly skewed variables"
            ])
        
        elif analysis_type == "correlation_analysis":
            if hasattr(results, 'average_correlation') and results.average_correlation > 0.7:
                recommendations.append("High correlations detected - consider dimensionality reduction")
            recommendations.extend([
                "Validate significant correlations with domain expertise",
                "Consider feature selection based on correlation findings"
            ])
        
        elif analysis_type == "hypothesis_testing":
            if results.get('p_value', 1) < 0.05:
                recommendations.append("Significant result found - consider practical significance")
            else:
                recommendations.append("No significant effect detected - consider sample size or alternative hypotheses")
        
        return recommendations
    
    def _assess_statistical_significance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess statistical significance across different test results."""
        significance_assessment = {
            "overall_significance": False,
            "significant_tests": [],
            "effect_sizes": {},
            "confidence_level": 0.95,
            "alpha_level": 0.05
        }
        
        # Check for p-values in results
        if 'p_value' in results:
            significance_assessment["overall_significance"] = results['p_value'] < 0.05
            if results['p_value'] < 0.05:
                significance_assessment["significant_tests"].append({
                    "test_name": results.get('test_type', 'Unknown'),
                    "p_value": results['p_value'],
                    "significance_level": "high" if results['p_value'] < 0.01 else "moderate"
                })
        
        return significance_assessment
    
    def _calculate_data_quality_score(self, results: Dict[str, Any]) -> float:
        """Calculate an overall data quality score."""
        # Simple quality score based on completeness and basic metrics
        dataset_summary = results.get('dataset_summary', {})
        total_cells = dataset_summary.get('total_rows', 1) * dataset_summary.get('total_columns', 1)
        missing_cells = dataset_summary.get('missing_values', {})
        total_missing = sum(missing_cells.values()) if isinstance(missing_cells, dict) else 0
        
        completeness_score = max(0, 1 - (total_missing / total_cells)) if total_cells > 0 else 0
        
        # Additional quality factors could be added here
        return completeness_score * 100  # Return as percentage