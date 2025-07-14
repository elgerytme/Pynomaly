"""
Comprehensive tests for Algorithm Optimization Service.
Tests algorithm parameter optimization, ensemble optimization, and adaptive configuration.
"""

import warnings
from unittest.mock import Mock, patch
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from pynomaly.application.services.algorithm_optimization_service import (
    AlgorithmOptimizationService,
)
from pynomaly.domain.entities import Dataset
from pynomaly.domain.entities.simple_detector import SimpleDetector


class TestAlgorithmOptimizationService:
    """Test suite for AlgorithmOptimizationService."""

    @pytest.fixture
    def optimization_service(self):
        """Algorithm optimization service."""
        return AlgorithmOptimizationService()

    @pytest.fixture
    def sample_detector(self):
        """Sample detector for testing."""
        return SimpleDetector(
            name="test_detector",
            algorithm_name="IsolationForest",
            parameters={"n_estimators": 100, "contamination": 0.1},
        )

    @pytest.fixture
    def sample_dataset(self):
        """Sample dataset for testing."""
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 1000),
                "feature2": np.random.normal(0, 1, 1000),
                "feature3": np.random.normal(0, 1, 1000),
            }
        )
        return Dataset(
            id=str(uuid4()),
            name="test_dataset",
            data=data,
        )

    @pytest.fixture
    def small_dataset(self):
        """Small dataset for testing size categorization."""
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 100),
                "feature2": np.random.normal(0, 1, 100),
            }
        )
        return Dataset(
            id=str(uuid4()),
            name="small_dataset",
            data=data,
        )

    @pytest.fixture
    def large_dataset(self):
        """Large dataset for testing size categorization."""
        np.random.seed(42)
        data = pd.DataFrame(
            {f"feature_{i}": np.random.normal(0, 1, 15000) for i in range(5)}
        )
        return Dataset(
            id=str(uuid4()),
            name="large_dataset",
            data=data,
        )

    @pytest.fixture
    def high_dim_dataset(self):
        """High-dimensional dataset for testing."""
        np.random.seed(42)
        data = pd.DataFrame(
            {f"feature_{i}": np.random.normal(0, 1, 500) for i in range(150)}
        )
        return Dataset(
            id=str(uuid4()),
            name="high_dim_dataset",
            data=data,
        )

    def test_service_initialization(self, optimization_service):
        """Test service initialization."""
        assert optimization_service is not None
        assert hasattr(optimization_service, "optimization_strategies")
        assert hasattr(optimization_service, "adaptive_parameters")
        assert hasattr(optimization_service, "optimization_cache")
        assert isinstance(optimization_service.optimization_cache, dict)

    def test_optimization_strategies_mapping(self, optimization_service):
        """Test that optimization strategies are properly mapped."""
        expected_algorithms = [
            "isolation_forest",
            "local_outlier_factor",
            "one_class_svm",
            "elliptic_envelope",
            "knn",
            "abod",
            "hbos",
            "pca",
        ]

        for algorithm in expected_algorithms:
            assert algorithm in optimization_service.optimization_strategies
            assert callable(optimization_service.optimization_strategies[algorithm])

    def test_dataset_characteristics_analysis_medium_dataset(
        self, optimization_service, sample_dataset
    ):
        """Test dataset characteristics analysis for medium dataset."""
        characteristics = optimization_service._analyze_dataset_characteristics(
            sample_dataset
        )

        assert characteristics["n_samples"] == 1000
        assert characteristics["n_features"] == 3
        assert characteristics["dataset_size_category"] == "medium"
        assert characteristics["dimensionality_category"] == "low"
        assert "density" in characteristics
        assert "mean_variance" in characteristics
        assert "feature_correlation" in characteristics
        assert "outlier_ratio_estimate" in characteristics
        assert "data_distribution" in characteristics
        assert "memory_footprint_mb" in characteristics
        assert "computational_complexity" in characteristics

    def test_dataset_characteristics_analysis_small_dataset(
        self, optimization_service, small_dataset
    ):
        """Test dataset characteristics analysis for small dataset."""
        characteristics = optimization_service._analyze_dataset_characteristics(
            small_dataset
        )

        assert characteristics["n_samples"] == 100
        assert characteristics["n_features"] == 2
        assert characteristics["dataset_size_category"] == "small"
        assert characteristics["dimensionality_category"] == "low"

    def test_dataset_characteristics_analysis_large_dataset(
        self, optimization_service, large_dataset
    ):
        """Test dataset characteristics analysis for large dataset."""
        characteristics = optimization_service._analyze_dataset_characteristics(
            large_dataset
        )

        assert characteristics["n_samples"] == 15000
        assert characteristics["n_features"] == 5
        assert characteristics["dataset_size_category"] == "large"
        assert characteristics["dimensionality_category"] == "low"

    def test_dataset_characteristics_analysis_high_dim(
        self, optimization_service, high_dim_dataset
    ):
        """Test dataset characteristics analysis for high-dimensional dataset."""
        characteristics = optimization_service._analyze_dataset_characteristics(
            high_dim_dataset
        )

        assert characteristics["n_samples"] == 500
        assert characteristics["n_features"] == 150
        assert characteristics["dataset_size_category"] == "small"
        assert characteristics["dimensionality_category"] == "high"

    @patch(
        "pynomaly.infrastructure.config.feature_flags.is_feature_enabled",
        return_value=True,
    )
    def test_optimize_detector_isolation_forest(
        self, mock_feature, optimization_service, sample_detector, sample_dataset
    ):
        """Test detector optimization for Isolation Forest."""
        optimized_detector, results = optimization_service.optimize_detector(
            sample_detector, sample_dataset, optimization_level="balanced"
        )

        assert optimized_detector is not None
        assert isinstance(optimized_detector, SimpleDetector)
        assert optimized_detector.name == f"{sample_detector.name}_optimized"
        assert optimized_detector.algorithm_name == sample_detector.algorithm_name
        assert optimized_detector.parameters is not None

        assert results is not None
        assert "best_params" in results
        assert "optimization_type" in results
        assert results["optimization_type"] == "heuristic_search"

    @patch(
        "pynomaly.infrastructure.config.feature_flags.is_feature_enabled",
        return_value=True,
    )
    def test_optimize_detector_different_levels(
        self, mock_feature, optimization_service, sample_detector, sample_dataset
    ):
        """Test detector optimization with different optimization levels."""
        levels = ["fast", "balanced", "thorough"]

        for level in levels:
            optimized_detector, results = optimization_service.optimize_detector(
                sample_detector, sample_dataset, optimization_level=level
            )

            assert optimized_detector is not None
            assert results is not None
            assert "best_params" in results

    @patch(
        "pynomaly.infrastructure.config.feature_flags.is_feature_enabled",
        return_value=True,
    )
    def test_optimize_detector_with_contamination_rate(
        self, mock_feature, optimization_service, sample_detector, sample_dataset
    ):
        """Test detector optimization with specific contamination rate."""
        contamination_rate = 0.05

        optimized_detector, results = optimization_service.optimize_detector(
            sample_detector,
            sample_dataset,
            contamination_rate=contamination_rate,
        )

        assert optimized_detector is not None
        assert results is not None
        assert "best_params" in results

        # Check that contamination rate was considered
        if "contamination" in results["best_params"]:
            assert results["best_params"]["contamination"] == contamination_rate

    @patch(
        "pynomaly.infrastructure.config.feature_flags.is_feature_enabled",
        return_value=True,
    )
    def test_optimize_detector_caching(
        self, mock_feature, optimization_service, sample_detector, sample_dataset
    ):
        """Test that optimization results are cached."""
        # First optimization
        _, results1 = optimization_service.optimize_detector(
            sample_detector, sample_dataset
        )

        # Second optimization (should use cache)
        _, results2 = optimization_service.optimize_detector(
            sample_detector, sample_dataset
        )

        # Results should be identical (from cache)
        assert results1 == results2

        # Cache should not be empty
        assert len(optimization_service.optimization_cache) > 0

    @patch(
        "pynomaly.infrastructure.config.feature_flags.is_feature_enabled",
        return_value=True,
    )
    def test_optimize_detector_unknown_algorithm(
        self, mock_feature, optimization_service, sample_dataset
    ):
        """Test optimization with unknown algorithm."""
        unknown_detector = SimpleDetector(
            name="unknown_detector",
            algorithm_name="UnknownAlgorithm",
            parameters={"param1": "value1"},
        )

        optimized_detector, results = optimization_service.optimize_detector(
            unknown_detector, sample_dataset
        )

        assert optimized_detector is not None
        assert results is not None
        assert results["optimization_type"] == "generic"

    @patch(
        "pynomaly.infrastructure.config.feature_flags.is_feature_enabled",
        return_value=True,
    )
    def test_optimize_ensemble(
        self, mock_feature, optimization_service, sample_dataset
    ):
        """Test ensemble optimization."""
        detectors = [
            SimpleDetector(
                name="detector1",
                algorithm_name="IsolationForest",
                parameters={"n_estimators": 100},
            ),
            SimpleDetector(
                name="detector2",
                algorithm_name="LocalOutlierFactor",
                parameters={"n_neighbors": 20},
            ),
        ]

        optimized_detectors, ensemble_results = optimization_service.optimize_ensemble(
            detectors, sample_dataset, ensemble_strategy="voting"
        )

        assert len(optimized_detectors) == len(detectors)
        assert all(isinstance(d, SimpleDetector) for d in optimized_detectors)

        assert ensemble_results is not None
        assert "strategy" in ensemble_results
        assert "n_detectors" in ensemble_results
        assert "combination_weights" in ensemble_results
        assert "individual_optimizations" in ensemble_results

        assert ensemble_results["strategy"] == "voting"
        assert ensemble_results["n_detectors"] == 2
        assert len(ensemble_results["combination_weights"]) == 2

    @patch(
        "pynomaly.infrastructure.config.feature_flags.is_feature_enabled",
        return_value=True,
    )
    def test_adaptive_parameter_selection(
        self, mock_feature, optimization_service, sample_dataset
    ):
        """Test adaptive parameter selection."""
        result = optimization_service.adaptive_parameter_selection(
            "IsolationForest", sample_dataset
        )

        assert result is not None
        assert "algorithm_name" in result
        assert "adaptive_parameters" in result
        assert "dataset_profile" in result
        assert "reasoning" in result

        assert result["algorithm_name"] == "IsolationForest"
        assert isinstance(result["adaptive_parameters"], dict)
        assert isinstance(result["reasoning"], list)
        assert len(result["reasoning"]) > 0

    @patch(
        "pynomaly.infrastructure.config.feature_flags.is_feature_enabled",
        return_value=True,
    )
    def test_adaptive_parameter_selection_with_performance_targets(
        self, mock_feature, optimization_service, sample_dataset
    ):
        """Test adaptive parameter selection with performance targets."""
        performance_targets = {"speed": 0.9, "accuracy": 0.8}

        result = optimization_service.adaptive_parameter_selection(
            "IsolationForest", sample_dataset, performance_targets=performance_targets
        )

        assert result is not None
        assert "adaptive_parameters" in result

        # Check that performance targets influenced parameters
        params = result["adaptive_parameters"]
        if "n_estimators" in params:
            # High speed requirement should reduce n_estimators
            assert params["n_estimators"] <= 100

    def test_isolation_forest_optimization_small_dataset(
        self, optimization_service, sample_detector, small_dataset
    ):
        """Test Isolation Forest optimization for small dataset."""
        profile = optimization_service._analyze_dataset_characteristics(small_dataset)

        result = optimization_service._optimize_isolation_forest(
            sample_detector,
            small_dataset,
            profile,
            "balanced",
            "cross_validation",
            None,
        )

        assert result is not None
        assert "best_params" in result
        assert "optimization_type" in result

        # Check adaptation for small dataset
        best_params = result["best_params"]
        if "n_estimators" in best_params:
            assert best_params["n_estimators"] <= 100

    def test_isolation_forest_optimization_large_dataset(
        self, optimization_service, sample_detector, large_dataset
    ):
        """Test Isolation Forest optimization for large dataset."""
        profile = optimization_service._analyze_dataset_characteristics(large_dataset)

        result = optimization_service._optimize_isolation_forest(
            sample_detector,
            large_dataset,
            profile,
            "balanced",
            "cross_validation",
            None,
        )

        assert result is not None
        assert "best_params" in result

        # Check adaptation for large dataset
        best_params = result["best_params"]
        if "n_estimators" in best_params:
            assert best_params["n_estimators"] >= 100

    def test_lof_optimization_high_dimensionality(
        self, optimization_service, sample_detector, high_dim_dataset
    ):
        """Test LOF optimization for high-dimensional dataset."""
        lof_detector = SimpleDetector(
            name="lof_detector",
            algorithm_name="LocalOutlierFactor",
            parameters={"n_neighbors": 20},
        )

        profile = optimization_service._analyze_dataset_characteristics(
            high_dim_dataset
        )

        result = optimization_service._optimize_local_outlier_factor(
            lof_detector,
            high_dim_dataset,
            profile,
            "balanced",
            "cross_validation",
            None,
        )

        assert result is not None
        assert "best_params" in result

        # Check adaptation for high dimensions
        best_params = result["best_params"]
        if "algorithm" in best_params:
            assert best_params["algorithm"] == "ball_tree"

    def test_one_class_svm_optimization_with_contamination(
        self, optimization_service, sample_detector, sample_dataset
    ):
        """Test One-Class SVM optimization with specific contamination rate."""
        svm_detector = SimpleDetector(
            name="svm_detector",
            algorithm_name="OneClassSVM",
            parameters={"nu": 0.1},
        )

        profile = optimization_service._analyze_dataset_characteristics(sample_dataset)
        contamination_rate = 0.08

        result = optimization_service._optimize_one_class_svm(
            svm_detector,
            sample_dataset,
            profile,
            "balanced",
            "cross_validation",
            contamination_rate,
        )

        assert result is not None
        assert "best_params" in result

        # Check that contamination rate influenced nu parameter
        best_params = result["best_params"]
        if "nu" in best_params:
            expected_range = [
                contamination_rate * 0.8,
                contamination_rate,
                contamination_rate * 1.2,
            ]
            assert best_params["nu"] in expected_range

    @patch(
        "pynomaly.infrastructure.config.feature_flags.is_feature_enabled",
        return_value=True,
    )
    @patch(
        "pynomaly.application.services.algorithm_benchmark.AlgorithmBenchmarkService"
    )
    def test_benchmark_optimization_impact(
        self,
        mock_benchmark_service,
        mock_feature,
        optimization_service,
        sample_detector,
        sample_dataset,
    ):
        """Test benchmarking optimization impact."""
        # Mock benchmark results
        mock_result = Mock()
        mock_result.execution_time = 1.0
        mock_result.memory_usage = 100.0
        mock_result.overall_score = Mock(return_value=0.8)
        mock_result.efficiency_score = Mock(return_value=0.7)

        mock_benchmark_service.return_value.benchmark_algorithm.return_value = [
            mock_result
        ]

        benchmark_results = optimization_service.benchmark_optimization_impact(
            sample_detector, sample_dataset, n_iterations=2
        )

        assert benchmark_results is not None
        assert "detector_name" in benchmark_results
        assert "algorithm_name" in benchmark_results
        assert "original_performance" in benchmark_results
        assert "optimized_performance" in benchmark_results
        assert "improvements" in benchmark_results
        assert "optimization_info" in benchmark_results
        assert "significant_improvement" in benchmark_results

        assert benchmark_results["detector_name"] == sample_detector.name
        assert benchmark_results["algorithm_name"] == sample_detector.algorithm_name

    def test_dataset_categorization_methods(self, optimization_service):
        """Test dataset categorization methods."""
        # Test size categorization
        assert optimization_service._categorize_dataset_size(500) == "small"
        assert optimization_service._categorize_dataset_size(5000) == "medium"
        assert optimization_service._categorize_dataset_size(15000) == "large"

        # Test dimensionality categorization
        assert optimization_service._categorize_dimensionality(5) == "low"
        assert optimization_service._categorize_dimensionality(50) == "medium"
        assert optimization_service._categorize_dimensionality(150) == "high"

    def test_outlier_ratio_estimation(self, optimization_service):
        """Test outlier ratio estimation using IQR method."""
        # Create data with known outliers
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, (900, 2))
        outliers = np.random.normal(5, 1, (100, 2))  # Clear outliers
        data = np.vstack([normal_data, outliers])

        outlier_ratio = optimization_service._estimate_outlier_ratio(data)

        # Should detect a reasonable portion of the outliers
        assert 0.05 <= outlier_ratio <= 0.3

    def test_data_distribution_analysis(self, optimization_service):
        """Test data distribution analysis."""
        # Centered and normalized data
        np.random.seed(42)
        centered_data = np.random.normal(0, 1, (1000, 3))

        distribution = optimization_service._analyze_data_distribution(centered_data)

        assert "is_centered" in distribution
        assert "is_normalized" in distribution
        assert "skewness" in distribution
        assert "kurtosis_estimate" in distribution

        # Should detect that data is approximately centered and normalized
        assert distribution["is_centered"] is True
        assert distribution["is_normalized"] is True

    def test_computational_complexity_estimation(self, optimization_service):
        """Test computational complexity estimation."""
        # Test different complexity levels
        assert optimization_service._estimate_computational_complexity(100, 10) == "low"
        assert (
            optimization_service._estimate_computational_complexity(1000, 100)
            == "medium"
        )
        assert (
            optimization_service._estimate_computational_complexity(10000, 200)
            == "high"
        )

    def test_algorithm_base_parameters(self, optimization_service):
        """Test getting base parameters for algorithms."""
        # Test known algorithms
        if_params = optimization_service._get_algorithm_base_parameters(
            "isolation_forest"
        )
        assert "n_estimators" in if_params
        assert "contamination" in if_params

        lof_params = optimization_service._get_algorithm_base_parameters(
            "local_outlier_factor"
        )
        assert "n_neighbors" in lof_params
        assert "contamination" in lof_params

        # Test unknown algorithm
        unknown_params = optimization_service._get_algorithm_base_parameters("unknown")
        assert "contamination" in unknown_params

    def test_dataset_adaptations(self, optimization_service):
        """Test dataset-specific parameter adaptations."""
        base_params = {"n_estimators": 100, "contamination": 0.1}

        # Test small dataset adaptations
        small_profile = {
            "dataset_size_category": "small",
            "dimensionality_category": "low",
        }
        adapted = optimization_service._apply_dataset_adaptations(
            base_params, small_profile, "isolation_forest"
        )
        assert isinstance(adapted, dict)

        # Test high-dimensional adaptations
        high_dim_profile = {
            "dataset_size_category": "medium",
            "dimensionality_category": "high",
        }
        adapted_high_dim = optimization_service._apply_dataset_adaptations(
            base_params, high_dim_profile, "isolation_forest"
        )
        assert isinstance(adapted_high_dim, dict)

    def test_performance_targets_application(self, optimization_service):
        """Test performance target adjustments."""
        base_params = {"n_estimators": 100, "max_samples": 0.8}

        # Test speed target
        speed_targets = {"speed": 0.9}
        speed_adjusted = optimization_service._apply_performance_targets(
            base_params, speed_targets, "isolation_forest"
        )
        assert speed_adjusted["n_estimators"] <= 100

        # Test accuracy target
        accuracy_targets = {"accuracy": 0.95}
        accuracy_adjusted = optimization_service._apply_performance_targets(
            base_params, accuracy_targets, "isolation_forest"
        )
        assert accuracy_adjusted["n_estimators"] >= 100

    def test_parameter_reasoning_generation(self, optimization_service):
        """Test parameter reasoning generation."""
        params = {"n_estimators": 150, "contamination": 0.1}
        profile = {
            "dataset_size_category": "medium",
            "dimensionality_category": "low",
            "n_samples": 5000,
            "n_features": 10,
        }

        reasoning = optimization_service._generate_parameter_reasoning(
            params, profile, "isolation_forest"
        )

        assert isinstance(reasoning, list)
        assert len(reasoning) > 0
        assert any("medium" in r for r in reasoning)
        assert any("5000" in r for r in reasoning)

    def test_limited_param_combinations_generation(self, optimization_service):
        """Test generation of limited parameter combinations."""
        param_grid = {
            "n_estimators": [50, 100, 200],
            "contamination": [0.05, 0.1, 0.15],
            "max_samples": [0.5, 0.8, "auto"],
        }

        # Test different optimization levels
        fast_combinations = optimization_service._generate_limited_param_combinations(
            param_grid, "fast"
        )
        assert len(fast_combinations) <= 10

        balanced_combinations = (
            optimization_service._generate_limited_param_combinations(
                param_grid, "balanced"
            )
        )
        assert len(balanced_combinations) <= 25

        thorough_combinations = (
            optimization_service._generate_limited_param_combinations(
                param_grid, "thorough"
            )
        )
        assert len(thorough_combinations) <= 50

        # Verify structure
        for combination in fast_combinations:
            assert isinstance(combination, dict)
            assert "n_estimators" in combination
            assert "contamination" in combination
            assert "max_samples" in combination

    def test_heuristic_parameter_selection(self, optimization_service, sample_dataset):
        """Test heuristic parameter selection."""
        param_combinations = [
            {"contamination": 0.05, "n_neighbors": 10, "n_estimators": 50},
            {"contamination": 0.1, "n_neighbors": 20, "n_estimators": 100},
            {"contamination": 0.15, "n_neighbors": 30, "n_estimators": 200},
        ]

        best_params = optimization_service._select_best_params_heuristic(
            param_combinations, sample_dataset, "isolation_forest"
        )

        assert isinstance(best_params, dict)
        assert "contamination" in best_params

        # For larger datasets, should prefer certain characteristics
        if sample_dataset.data.shape[0] > 1000:
            if "contamination" in best_params:
                assert best_params["contamination"] <= 0.1

    def test_cache_key_generation(self, optimization_service, sample_dataset):
        """Test cache key generation for optimization results."""
        cache_key = optimization_service._generate_cache_key(
            sample_dataset, "IsolationForest", "balanced"
        )

        assert isinstance(cache_key, str)
        assert "IsolationForest" in cache_key
        assert "balanced" in cache_key

        # Different datasets should generate different keys
        different_dataset = Dataset(
            id=str(uuid4()),
            name="different",
            data=pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}),
        )

        different_key = optimization_service._generate_cache_key(
            different_dataset, "IsolationForest", "balanced"
        )

        assert cache_key != different_key

    def test_optimization_with_warnings(
        self, optimization_service, sample_detector, sample_dataset
    ):
        """Test that optimization handles warnings gracefully."""
        # Patch to simulate optimization failure
        with patch.object(
            optimization_service,
            "_perform_parameter_search",
            side_effect=Exception("Test error"),
        ):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")

                result = optimization_service._optimize_isolation_forest(
                    sample_detector,
                    sample_dataset,
                    {},
                    "balanced",
                    "cross_validation",
                    None,
                )

                # Should return default parameters
                assert result is not None
                assert result["optimization_type"] == "default"

                # Should have issued a warning
                assert len(w) > 0
                assert "Optimization failed" in str(w[0].message)

    def test_pyod_algorithm_optimization_with_import_error(
        self, optimization_service, sample_detector, sample_dataset
    ):
        """Test PyOD algorithm optimization when imports fail."""
        profile = optimization_service._analyze_dataset_characteristics(sample_dataset)

        # Test KNN optimization with import error
        knn_detector = SimpleDetector(
            name="knn_detector",
            algorithm_name="KNN",
            parameters={"n_neighbors": 5},
        )

        with patch(
            "pynomaly.application.services.algorithm_optimization_service.PyODAdapter",
            side_effect=ImportError,
        ):
            result = optimization_service._optimize_knn(
                knn_detector,
                sample_dataset,
                profile,
                "balanced",
                "cross_validation",
                None,
            )

            assert result is not None
            assert result["optimization_type"] == "default"

    def test_ensemble_combination_optimization(
        self, optimization_service, sample_dataset
    ):
        """Test ensemble combination optimization."""
        detectors = [
            SimpleDetector(name="d1", algorithm_name="IsolationForest", parameters={}),
            SimpleDetector(name="d2", algorithm_name="LOF", parameters={}),
        ]

        individual_results = [
            {"best_score": 0.8},
            {"best_score": 0.7},
        ]

        result = optimization_service._optimize_ensemble_combination(
            detectors, sample_dataset, "voting", individual_results
        )

        assert result is not None
        assert "strategy" in result
        assert "n_detectors" in result
        assert "combination_weights" in result

        assert result["strategy"] == "voting"
        assert result["n_detectors"] == 2
        assert len(result["combination_weights"]) == 2

        # Weights should be proportional to performance
        weights = result["combination_weights"]
        assert weights[0] > weights[1]  # Higher score should have higher weight

    def test_improvement_calculations(self, optimization_service):
        """Test optimization improvement calculations."""
        # Mock results
        original_results = []
        optimized_results = []

        for i in range(3):
            orig_result = Mock()
            orig_result.execution_time = 2.0
            orig_result.memory_usage = 200.0
            orig_result.overall_score = Mock(return_value=0.7)
            original_results.append(orig_result)

            opt_result = Mock()
            opt_result.execution_time = 1.5
            opt_result.memory_usage = 150.0
            opt_result.overall_score = Mock(return_value=0.8)
            optimized_results.append(opt_result)

        improvements = optimization_service._calculate_optimization_improvements(
            original_results, optimized_results
        )

        assert "execution_time_improvement" in improvements
        assert "memory_improvement" in improvements
        assert "accuracy_improvement" in improvements
        assert "overall_improvement" in improvements

        # Should show improvements
        assert improvements["execution_time_improvement"] > 0
        assert improvements["memory_improvement"] > 0
        assert improvements["accuracy_improvement"] > 0

    def test_benchmark_results_aggregation(self, optimization_service):
        """Test benchmark results aggregation."""
        results = []
        for i in range(3):
            result = Mock()
            result.execution_time = 1.0 + i * 0.1
            result.memory_usage = 100.0 + i * 10
            result.overall_score = Mock(return_value=0.8 + i * 0.05)
            result.efficiency_score = Mock(return_value=0.7 + i * 0.05)
            results.append(result)

        aggregated = optimization_service._aggregate_benchmark_results(results)

        assert "avg_execution_time" in aggregated
        assert "avg_memory_usage" in aggregated
        assert "avg_overall_score" in aggregated
        assert "avg_efficiency_score" in aggregated

        # Check averages are calculated correctly
        assert abs(aggregated["avg_execution_time"] - 1.1) < 0.01
        assert abs(aggregated["avg_memory_usage"] - 110.0) < 0.01

    @patch(
        "pynomaly.infrastructure.config.feature_flags.is_feature_enabled",
        return_value=False,
    )
    def test_feature_flag_disabled(
        self, mock_feature, optimization_service, sample_detector, sample_dataset
    ):
        """Test behavior when feature flag is disabled."""
        with pytest.raises(Exception):  # Should raise when feature is disabled
            optimization_service.optimize_detector(sample_detector, sample_dataset)
