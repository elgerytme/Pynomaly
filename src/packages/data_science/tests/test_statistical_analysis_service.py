"""Comprehensive tests for statistical analysis service implementation."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from datetime import datetime

from ..infrastructure.services.statistical_analysis_service_impl import StatisticalAnalysisServiceImpl
from ..domain.entities.analysis_job import AnalysisJob, AnalysisJobStatus
from ..domain.value_objects.statistical_metrics import StatisticalMetrics
from ..domain.value_objects.correlation_matrix import CorrelationMatrix, CorrelationType


@pytest.fixture
def statistical_service():
    """Create statistical analysis service instance."""
    return StatisticalAnalysisServiceImpl()


@pytest.fixture
def sample_numeric_data():
    """Create sample numeric dataset for testing."""
    np.random.seed(42)
    data = {
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.normal(50, 10, 1000),
        'feature3': np.random.uniform(0, 100, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000),
        'outlier_feature': np.concatenate([
            np.random.normal(10, 2, 990),
            np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])  # Outliers
        ])
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_time_series_data():
    """Create sample time series data for testing."""
    dates = pd.date_range('2023-01-01', periods=365, freq='D')
    values = 100 + np.cumsum(np.random.normal(0, 1, 365)) + 10 * np.sin(2 * np.pi * np.arange(365) / 30)
    return pd.DataFrame({'timestamp': dates, 'value': values})


@pytest.fixture
def sample_analysis_job():
    """Create sample analysis job for testing."""
    return AnalysisJob(
        job_name="Test Analysis",
        analysis_type="descriptive_statistics",
        dataset_id="test_dataset_123",
        status=AnalysisJobStatus.COMPLETED,
        created_at=datetime.utcnow(),
        completed_at=datetime.utcnow()
    )


class TestStatisticalAnalysisService:
    """Test suite for statistical analysis service."""

    @pytest.mark.asyncio
    async def test_perform_descriptive_analysis(self, statistical_service, sample_numeric_data):
        """Test descriptive statistical analysis."""
        config = {"include_all_stats": True}
        
        result = await statistical_service.perform_descriptive_analysis(sample_numeric_data, config)
        
        assert "descriptive_statistics" in result
        assert "dataset_summary" in result
        
        # Check dataset summary
        summary = result["dataset_summary"]
        assert summary["total_rows"] == 1000
        assert summary["total_columns"] == 5
        assert summary["numeric_columns"] == 4
        
        # Check descriptive stats for numeric columns
        desc_stats = result["descriptive_statistics"]
        assert "feature1" in desc_stats
        assert "feature2" in desc_stats
        
        # Verify statistical measures
        feature1_stats = desc_stats["feature1"]
        assert "mean" in feature1_stats
        assert "std" in feature1_stats
        assert "min" in feature1_stats
        assert "max" in feature1_stats
        assert "median" in feature1_stats
        assert "skewness" in feature1_stats
        assert "kurtosis" in feature1_stats

    @pytest.mark.asyncio
    async def test_perform_correlation_analysis_pearson(self, statistical_service, sample_numeric_data):
        """Test Pearson correlation analysis."""
        features = ['feature1', 'feature2', 'feature3']
        
        result = await statistical_service.perform_correlation_analysis(
            sample_numeric_data, features, method="pearson"
        )
        
        assert isinstance(result, CorrelationMatrix)
        assert result.correlation_type == CorrelationType.PEARSON
        assert len(result.features) == 3
        assert len(result.correlation_matrix) == 3
        assert len(result.correlation_matrix[0]) == 3
        assert result.significance_level == 0.05
        assert result.determinant is not None
        assert result.condition_index is not None

    @pytest.mark.asyncio
    async def test_perform_correlation_analysis_spearman(self, statistical_service, sample_numeric_data):
        """Test Spearman correlation analysis."""
        features = ['feature1', 'feature2']
        
        result = await statistical_service.perform_correlation_analysis(
            sample_numeric_data, features, method="spearman"
        )
        
        assert isinstance(result, CorrelationMatrix)
        assert result.correlation_type == CorrelationType.SPEARMAN
        assert len(result.features) == 2

    @pytest.mark.asyncio
    async def test_analyze_data_distribution(self, statistical_service, sample_numeric_data):
        """Test data distribution analysis."""
        feature = 'feature1'
        
        result = await statistical_service.analyze_data_distribution(sample_numeric_data, feature)
        
        assert result.feature_name == feature
        assert result.sample_size == 1000
        assert "mu" in result.parameters
        assert "sigma" in result.parameters
        assert len(result.normality_tests) >= 2
        assert result.goodness_of_fit_score is not None
        assert isinstance(result.is_normal, bool)
        
        # Check normality tests
        shapiro_test = next((test for test in result.normality_tests if test.test_name == "Shapiro-Wilk"), None)
        assert shapiro_test is not None
        assert shapiro_test.p_value is not None
        assert shapiro_test.interpretation in ["Normal", "Not normal"]

    @pytest.mark.asyncio
    async def test_detect_outliers_iqr_method(self, statistical_service, sample_numeric_data):
        """Test outlier detection using IQR method."""
        features = ['outlier_feature']
        methods = ['iqr']
        
        result = await statistical_service.detect_outliers(sample_numeric_data, features, methods)
        
        assert 'outlier_feature' in result
        assert 'iqr' in result['outlier_feature']
        
        iqr_results = result['outlier_feature']['iqr']
        assert iqr_results['outlier_count'] > 0  # Should detect the inserted outliers
        assert iqr_results['outlier_percentage'] > 0
        assert 'lower_bound' in iqr_results
        assert 'upper_bound' in iqr_results
        assert len(iqr_results['outlier_indices']) == iqr_results['outlier_count']

    @pytest.mark.asyncio
    async def test_detect_outliers_zscore_method(self, statistical_service, sample_numeric_data):
        """Test outlier detection using Z-score method."""
        features = ['outlier_feature']
        methods = ['zscore']
        
        result = await statistical_service.detect_outliers(sample_numeric_data, features, methods)
        
        assert 'outlier_feature' in result
        assert 'zscore' in result['outlier_feature']
        
        zscore_results = result['outlier_feature']['zscore']
        assert zscore_results['outlier_count'] > 0
        assert zscore_results['threshold'] == 3.0

    @pytest.mark.asyncio
    async def test_detect_outliers_isolation_forest(self, statistical_service, sample_numeric_data):
        """Test outlier detection using Isolation Forest."""
        features = ['outlier_feature']
        methods = ['isolation_forest']
        
        result = await statistical_service.detect_outliers(sample_numeric_data, features, methods)
        
        assert 'outlier_feature' in result
        assert 'isolation_forest' in result['outlier_feature']
        
        if 'error' not in result['outlier_feature']['isolation_forest']:
            isolation_results = result['outlier_feature']['isolation_forest']
            assert 'outlier_count' in isolation_results
            assert 'contamination' in isolation_results
            assert isolation_results['contamination'] == 0.1

    @pytest.mark.asyncio
    async def test_detect_outliers_modified_zscore(self, statistical_service, sample_numeric_data):
        """Test outlier detection using Modified Z-score (MAD-based)."""
        features = ['outlier_feature']
        methods = ['modified_zscore']
        
        result = await statistical_service.detect_outliers(sample_numeric_data, features, methods)
        
        assert 'outlier_feature' in result
        assert 'modified_zscore' in result['outlier_feature']
        
        mad_results = result['outlier_feature']['modified_zscore']
        assert 'median_absolute_deviation' in mad_results
        assert mad_results['threshold'] == 3.5

    @pytest.mark.asyncio
    async def test_hypothesis_test_one_sample_t_test(self, statistical_service, sample_numeric_data):
        """Test one-sample t-test."""
        test_config = {
            "column": "feature1",
            "null_mean": 100
        }
        
        result = await statistical_service.perform_hypothesis_test(
            sample_numeric_data, "t_test_one_sample", test_config
        )
        
        assert result["test_type"] == "One-sample t-test"
        assert "statistic" in result
        assert "p_value" in result
        assert result["null_hypothesis"] == "Mean equals 100"
        assert result["interpretation"] in ["Reject null hypothesis", "Fail to reject null hypothesis"]

    @pytest.mark.asyncio
    async def test_hypothesis_test_independent_t_test(self, statistical_service):
        """Test independent samples t-test."""
        # Create data with two groups
        group_data = pd.DataFrame({
            'value': np.concatenate([np.random.normal(50, 10, 100), np.random.normal(55, 10, 100)]),
            'group': ['A'] * 100 + ['B'] * 100
        })
        
        test_config = {
            "group_column": "group",
            "value_column": "value",
            "group1": "A",
            "group2": "B"
        }
        
        result = await statistical_service.perform_hypothesis_test(
            group_data, "t_test_independent", test_config
        )
        
        assert result["test_type"] == "Independent samples t-test"
        assert "effect_size" in result
        assert result["null_hypothesis"] == "Group means are equal"

    @pytest.mark.asyncio
    async def test_hypothesis_test_paired_t_test(self, statistical_service):
        """Test paired samples t-test."""
        # Create paired data
        before_values = np.random.normal(50, 10, 100)
        after_values = before_values + np.random.normal(5, 3, 100)  # Add effect
        
        paired_data = pd.DataFrame({
            'before': before_values,
            'after': after_values
        })
        
        test_config = {
            "before_column": "before",
            "after_column": "after"
        }
        
        result = await statistical_service.perform_hypothesis_test(
            paired_data, "t_test_paired", test_config
        )
        
        assert result["test_type"] == "Paired samples t-test"
        assert "mean_difference" in result
        assert "pairs_count" in result
        assert result["pairs_count"] == 100

    @pytest.mark.asyncio
    async def test_hypothesis_test_chi_square_independence(self, statistical_service):
        """Test chi-square test of independence."""
        # Create categorical data
        cat_data = pd.DataFrame({
            'variable1': np.random.choice(['X', 'Y'], 200),
            'variable2': np.random.choice(['P', 'Q', 'R'], 200)
        })
        
        test_config = {
            "variable1": "variable1",
            "variable2": "variable2"
        }
        
        result = await statistical_service.perform_hypothesis_test(
            cat_data, "chi_square_independence", test_config
        )
        
        assert result["test_type"] == "Chi-square test of independence"
        assert "cramers_v" in result
        assert "degrees_of_freedom" in result
        assert "contingency_table" in result

    @pytest.mark.asyncio
    async def test_hypothesis_test_anova_one_way(self, statistical_service):
        """Test one-way ANOVA."""
        # Create data with three groups
        group_data = pd.DataFrame({
            'value': np.concatenate([
                np.random.normal(50, 10, 50),
                np.random.normal(55, 10, 50),
                np.random.normal(60, 10, 50)
            ]),
            'group': ['A'] * 50 + ['B'] * 50 + ['C'] * 50
        })
        
        test_config = {
            "groups": ["A", "B", "C"],
            "value_column": "value",
            "group_column": "group"
        }
        
        result = await statistical_service.perform_hypothesis_test(
            group_data, "anova_one_way", test_config
        )
        
        assert result["test_type"] == "One-way ANOVA"
        assert "eta_squared" in result
        assert "group_means" in result
        assert len(result["group_means"]) == 3
        assert "degrees_of_freedom_between" in result
        assert "degrees_of_freedom_within" in result

    @pytest.mark.asyncio
    async def test_hypothesis_test_mann_whitney_u(self, statistical_service):
        """Test Mann-Whitney U test."""
        # Create non-normal data
        group_data = pd.DataFrame({
            'value': np.concatenate([
                np.random.exponential(2, 100),
                np.random.exponential(3, 100)
            ]),
            'group': ['A'] * 100 + ['B'] * 100
        })
        
        test_config = {
            "group_column": "group",
            "value_column": "value",
            "group1": "A",
            "group2": "B"
        }
        
        result = await statistical_service.perform_hypothesis_test(
            group_data, "mann_whitney_u", test_config
        )
        
        assert result["test_type"] == "Mann-Whitney U test"
        assert "group1_median" in result
        assert "group2_median" in result

    @pytest.mark.asyncio
    async def test_calculate_feature_statistics_numeric(self, statistical_service, sample_numeric_data):
        """Test comprehensive feature statistics for numeric data."""
        feature = 'feature1'
        
        result = await statistical_service.calculate_feature_statistics(sample_numeric_data, feature)
        
        assert isinstance(result, StatisticalMetrics)
        assert result.feature_name == feature
        assert result.sample_size == 1000
        assert result.data_type == 'float64'
        assert result.mean is not None
        assert result.median is not None
        assert result.standard_deviation is not None
        assert result.variance is not None
        assert result.skewness is not None
        assert result.kurtosis is not None
        assert result.unique_count is not None
        assert result.outlier_count >= 0

    @pytest.mark.asyncio
    async def test_calculate_feature_statistics_categorical(self, statistical_service, sample_numeric_data):
        """Test comprehensive feature statistics for categorical data."""
        feature = 'category'
        
        result = await statistical_service.calculate_feature_statistics(sample_numeric_data, feature)
        
        assert isinstance(result, StatisticalMetrics)
        assert result.feature_name == feature
        assert result.sample_size == 1000
        assert result.unique_count == 3  # A, B, C
        assert result.entropy is not None
        assert result.mean is None  # No mean for categorical
        assert result.standard_deviation is None

    @pytest.mark.asyncio
    async def test_compare_distributions_numeric(self, statistical_service):
        """Test distribution comparison for numeric data."""
        # Create two datasets with different distributions
        dataset1 = pd.DataFrame({'feature': np.random.normal(50, 10, 200)})
        dataset2 = pd.DataFrame({'feature': np.random.normal(55, 12, 200)})
        
        result = await statistical_service.compare_distributions(dataset1, dataset2, 'feature')
        
        assert result["feature"] == "feature"
        assert result["dataset1_size"] == 200
        assert result["dataset2_size"] == 200
        assert "tests" in result
        assert "effect_sizes" in result
        assert "descriptive_comparison" in result
        
        # Check location test (t-test or Mann-Whitney)
        assert "location_test" in result["tests"]
        location_test = result["tests"]["location_test"]
        assert location_test["test_name"] in ["Independent t-test", "Mann-Whitney U test"]
        
        # Check distribution test (Kolmogorov-Smirnov)
        assert "distribution_test" in result["tests"]
        ks_test = result["tests"]["distribution_test"]
        assert ks_test["test_name"] == "Kolmogorov-Smirnov"
        
        # Check effect sizes
        assert "cohens_d" in result["effect_sizes"]
        cohens_d = result["effect_sizes"]["cohens_d"]
        assert "value" in cohens_d
        assert "magnitude" in cohens_d

    @pytest.mark.asyncio
    async def test_compare_distributions_categorical(self, statistical_service):
        """Test distribution comparison for categorical data."""
        # Create two datasets with different categorical distributions
        dataset1 = pd.DataFrame({'feature': np.random.choice(['A', 'B', 'C'], 200, p=[0.5, 0.3, 0.2])})
        dataset2 = pd.DataFrame({'feature': np.random.choice(['A', 'B', 'C'], 200, p=[0.3, 0.4, 0.3])})
        
        result = await statistical_service.compare_distributions(dataset1, dataset2, 'feature')
        
        assert "tests" in result
        assert "independence_test" in result["tests"]
        
        chi2_test = result["tests"]["independence_test"]
        assert chi2_test["test_name"] == "Chi-square test of independence"
        assert "degrees_of_freedom" in chi2_test
        
        # Check Cramér's V effect size
        assert "effect_sizes" in result
        assert "cramers_v" in result["effect_sizes"]

    @pytest.mark.asyncio
    async def test_analyze_time_series(self, statistical_service, sample_time_series_data):
        """Test time series analysis."""
        config = {"detect_seasonality": True}
        
        result = await statistical_service.analyze_time_series(
            sample_time_series_data, "timestamp", "value", config
        )
        
        assert "time_range" in result
        assert "descriptive_stats" in result
        assert "autocorrelation" in result
        assert "trend_analysis" in result
        
        # Check time range
        time_range = result["time_range"]
        assert "start" in time_range
        assert "end" in time_range
        assert time_range["observations"] == 365
        
        # Check trend analysis
        trend_analysis = result["trend_analysis"]
        assert "linear_slope" in trend_analysis
        assert "r_squared" in trend_analysis
        assert "significant_trend" in trend_analysis
        assert trend_analysis["trend_interpretation"] in [
            "no_significant_trend", "significant_upward_trend", "significant_downward_trend"
        ]
        
        # Check autocorrelation
        autocorr = result["autocorrelation"]
        assert "lag_1" in autocorr

    @pytest.mark.asyncio
    async def test_validate_analysis_requirements_valid_data(self, statistical_service, sample_numeric_data):
        """Test analysis requirements validation with valid data."""
        result = await statistical_service.validate_analysis_requirements(
            sample_numeric_data, "correlation_analysis"
        )
        
        assert result["analysis_type"] == "correlation_analysis"
        assert result["dataset_valid"] is True
        assert result["requirements_met"] is True
        assert len(result["errors"]) == 0
        
        # Check dataset info
        dataset_info = result["dataset_info"]
        assert dataset_info["shape"] == (1000, 5)
        assert "memory_usage_mb" in dataset_info
        assert "missing_values_total" in dataset_info

    @pytest.mark.asyncio
    async def test_validate_analysis_requirements_insufficient_columns(self, statistical_service):
        """Test validation with insufficient numeric columns for correlation."""
        # Create dataset with only one numeric column
        data = pd.DataFrame({
            'feature1': np.random.normal(50, 10, 100),
            'category': ['A', 'B', 'C'] * 33 + ['A']
        })
        
        result = await statistical_service.validate_analysis_requirements(data, "correlation_analysis")
        
        assert result["requirements_met"] is False
        assert len(result["errors"]) > 0
        assert any("at least 2 numeric columns" in error for error in result["errors"])

    @pytest.mark.asyncio
    async def test_validate_analysis_requirements_empty_dataset(self, statistical_service):
        """Test validation with empty dataset."""
        empty_data = pd.DataFrame()
        
        result = await statistical_service.validate_analysis_requirements(empty_data, "descriptive_statistics")
        
        assert result["dataset_valid"] is False
        assert result["requirements_met"] is False
        assert len(result["errors"]) > 0
        assert any("empty" in error.lower() for error in result["errors"])

    @pytest.mark.asyncio
    async def test_generate_analysis_report(self, statistical_service, sample_analysis_job):
        """Test comprehensive analysis report generation."""
        # Mock analysis results
        results = {
            "descriptive_statistics": {
                "feature1": {"mean": 100.5, "std": 15.2}
            },
            "dataset_summary": {
                "total_rows": 1000,
                "total_columns": 5,
                "numeric_columns": 4
            }
        }
        
        report = await statistical_service.generate_analysis_report(sample_analysis_job, results)
        
        assert "report_id" in report
        assert "generated_at" in report
        assert "analysis_job" in report
        assert "executive_summary" in report
        assert "detailed_results" in report
        assert "recommendations" in report
        assert "statistical_significance" in report
        
        # Check analysis job info
        job_info = report["analysis_job"]
        assert job_info["job_name"] == "Test Analysis"
        assert job_info["analysis_type"] == "descriptive_statistics"
        
        # Check executive summary
        exec_summary = report["executive_summary"]
        assert exec_summary["summary_type"] == "descriptive_statistics"
        assert "quality_score" in exec_summary
        
        # Check recommendations
        assert len(report["recommendations"]) > 0

    @pytest.mark.asyncio
    async def test_error_handling_invalid_feature(self, statistical_service, sample_numeric_data):
        """Test error handling for invalid feature names."""
        with pytest.raises(ValueError, match="Feature 'nonexistent' not found"):
            await statistical_service.calculate_feature_statistics(sample_numeric_data, 'nonexistent')

    @pytest.mark.asyncio
    async def test_error_handling_unsupported_test_type(self, statistical_service, sample_numeric_data):
        """Test error handling for unsupported hypothesis test types."""
        with pytest.raises(ValueError, match="Unsupported test type"):
            await statistical_service.perform_hypothesis_test(
                sample_numeric_data, "unsupported_test", {}
            )

    @pytest.mark.asyncio
    async def test_error_handling_unsupported_correlation_method(self, statistical_service, sample_numeric_data):
        """Test error handling for unsupported correlation methods."""
        with pytest.raises(ValueError, match="Unsupported correlation method"):
            await statistical_service.perform_correlation_analysis(
                sample_numeric_data, None, method="unsupported"
            )

    @pytest.mark.asyncio
    async def test_error_handling_non_dataframe_input(self, statistical_service):
        """Test error handling for non-DataFrame inputs."""
        with pytest.raises(NotImplementedError, match="Only pandas DataFrame currently supported"):
            await statistical_service.perform_descriptive_analysis("not_a_dataframe", {})

    def test_interpret_cohens_d(self, statistical_service):
        """Test Cohen's d interpretation."""
        assert statistical_service._interpret_cohens_d(0.1) == "negligible"
        assert statistical_service._interpret_cohens_d(0.3) == "small"
        assert statistical_service._interpret_cohens_d(0.6) == "medium"
        assert statistical_service._interpret_cohens_d(1.0) == "large"
        assert statistical_service._interpret_cohens_d(-0.7) == "medium"

    def test_interpret_cramers_v(self, statistical_service):
        """Test Cramér's V interpretation."""
        assert statistical_service._interpret_cramers_v(0.05) == "negligible"
        assert statistical_service._interpret_cramers_v(0.2) == "small"
        assert statistical_service._interpret_cramers_v(0.4) == "medium"
        assert statistical_service._interpret_cramers_v(0.6) == "large"

    def test_determine_trend_direction(self, statistical_service):
        """Test trend direction determination."""
        # Increasing trend
        increasing_series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        assert statistical_service._determine_trend_direction(increasing_series) == "increasing"
        
        # Decreasing trend
        decreasing_series = pd.Series([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
        assert statistical_service._determine_trend_direction(decreasing_series) == "decreasing"
        
        # Stable trend
        stable_series = pd.Series([5, 5, 5, 5, 5, 5, 5, 5, 5, 5])
        assert statistical_service._determine_trend_direction(stable_series) == "stable"

    def test_interpret_trend(self, statistical_service):
        """Test trend interpretation."""
        assert statistical_service._interpret_trend(0.5, 0.02) == "significant_upward_trend"
        assert statistical_service._interpret_trend(-0.5, 0.02) == "significant_downward_trend"
        assert statistical_service._interpret_trend(0.5, 0.1) == "no_significant_trend"

    def test_calculate_data_quality_score(self, statistical_service):
        """Test data quality score calculation."""
        # Perfect data (no missing values)
        results_perfect = {
            "dataset_summary": {
                "total_rows": 100,
                "total_columns": 5,
                "missing_values": {"col1": 0, "col2": 0, "col3": 0, "col4": 0, "col5": 0}
            }
        }
        score_perfect = statistical_service._calculate_data_quality_score(results_perfect)
        assert score_perfect == 100.0
        
        # Data with some missing values
        results_missing = {
            "dataset_summary": {
                "total_rows": 100,
                "total_columns": 4,
                "missing_values": {"col1": 0, "col2": 10, "col3": 5, "col4": 5}  # 20 missing out of 400
            }
        }
        score_missing = statistical_service._calculate_data_quality_score(results_missing)
        assert score_missing == 95.0  # (400-20)/400 * 100


class TestStatisticalAnalysisServiceEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_small_sample_correlation(self, statistical_service):
        """Test correlation analysis with very small sample."""
        small_data = pd.DataFrame({
            'x': [1, 2, 3],
            'y': [2, 4, 6]
        })
        
        result = await statistical_service.perform_correlation_analysis(small_data, None, "pearson")
        assert isinstance(result, CorrelationMatrix)
        assert len(result.features) == 2

    @pytest.mark.asyncio
    async def test_constant_feature_statistics(self, statistical_service):
        """Test statistics for constant feature (zero variance)."""
        constant_data = pd.DataFrame({'constant': [5] * 100})
        
        result = await statistical_service.calculate_feature_statistics(constant_data, 'constant')
        assert result.variance == 0
        assert result.standard_deviation == 0
        assert result.coefficient_of_variation is None  # Division by zero handled

    @pytest.mark.asyncio
    async def test_all_missing_values_feature(self, statistical_service):
        """Test feature with all missing values."""
        missing_data = pd.DataFrame({'missing': [np.nan] * 100})
        
        result = await statistical_service.calculate_feature_statistics(missing_data, 'missing')
        assert result.sample_size == 0  # After dropna()
        assert result.missing_count == 100
        assert result.missing_percentage == 100.0

    @pytest.mark.asyncio
    async def test_single_category_distribution(self, statistical_service):
        """Test distribution analysis with single category."""
        single_cat_data = pd.DataFrame({'category': ['A'] * 100})
        
        result = await statistical_service.calculate_feature_statistics(single_cat_data, 'category')
        assert result.unique_count == 1
        assert result.entropy == 0  # Perfect certainty

    @pytest.mark.asyncio
    async def test_time_series_insufficient_data(self, statistical_service):
        """Test time series analysis with insufficient data points."""
        small_ts_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=2),
            'value': [1, 2]
        })
        
        result = await statistical_service.analyze_time_series(
            small_ts_data, "timestamp", "value", {}
        )
        
        assert result["time_range"]["observations"] == 2
        # Should still provide basic analysis even with minimal data
        assert "trend_analysis" in result