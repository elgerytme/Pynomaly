"""Tests for intelligent algorithm selection service."""

from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from pynomaly.application.dto.selection_dto import (
    AlgorithmPerformanceDTO,
    DatasetCharacteristicsDTO,
    OptimizationConstraintsDTO,
    SelectionRecommendationDTO,
)
from pynomaly.application.services.intelligent_selection_service import (
    IntelligentSelectionService,
)
from pynomaly.domain.entities import Dataset


class TestIntelligentSelectionService:
    """Test intelligent selection service."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create test data directory
        self.test_data_dir = Path("/tmp/pynomaly_test")
        self.test_data_dir.mkdir(exist_ok=True)

        # Initialize service
        self.service = IntelligentSelectionService(
            enable_meta_learning=True,
            enable_performance_prediction=True,
            enable_historical_learning=True,
            selection_history_path=self.test_data_dir / "history.json",
            meta_model_path=self.test_data_dir / "models",
        )

        # Create test dataset
        np.random.seed(42)
        self.test_data = pd.DataFrame(
            {
                "feature_1": np.random.normal(0, 1, 1000),
                "feature_2": np.random.normal(0, 1, 1000),
                "feature_3": np.random.normal(0, 1, 1000),
                "feature_4": np.random.choice(["A", "B", "C"], 1000),
            }
        )

        self.test_dataset = Dataset(
            name="test_dataset",
            data=self.test_data,
            feature_names=["feature_1", "feature_2", "feature_3", "feature_4"],
        )

    def test_initialization(self):
        """Test service initialization."""
        assert self.service is not None
        assert self.service.enable_meta_learning is True
        assert self.service.enable_performance_prediction is True
        assert self.service.enable_historical_learning is True
        assert isinstance(self.service.algorithm_registry, dict)
        assert len(self.service.algorithm_registry) > 0

    def test_initialization_with_options(self):
        """Test service initialization with custom options."""
        service = IntelligentSelectionService(
            enable_meta_learning=False,
            enable_performance_prediction=False,
            enable_historical_learning=False,
        )

        assert service.enable_meta_learning is False
        assert service.enable_performance_prediction is False
        assert service.enable_historical_learning is False

    @pytest.mark.asyncio
    async def test_extract_dataset_characteristics(self):
        """Test dataset characteristics extraction."""
        characteristics = await self.service._extract_dataset_characteristics(
            self.test_dataset
        )

        assert isinstance(characteristics, DatasetCharacteristicsDTO)
        assert characteristics.n_samples == 1000
        assert characteristics.n_features == 4
        assert characteristics.n_numeric_features == 3
        assert characteristics.n_categorical_features == 1
        assert 0 <= characteristics.feature_density <= 1
        assert 0 <= characteristics.outlier_ratio <= 1
        assert 0 <= characteristics.mean_feature_correlation <= 1

    def test_filter_algorithm_candidates(self):
        """Test algorithm candidate filtering."""
        characteristics = DatasetCharacteristicsDTO(
            n_samples=1000,
            n_features=4,
            n_numeric_features=3,
            n_categorical_features=1,
            feature_density=0.9,
            missing_value_ratio=0.0,
            outlier_ratio=0.05,
            mean_feature_correlation=0.1,
            feature_variance_ratio=0.2,
            data_dimensionality_ratio=0.004,
            skewness_mean=0.1,
            kurtosis_mean=0.1,
            class_imbalance=0.0,
        )

        # Test without constraints
        candidates = self.service._filter_algorithm_candidates(characteristics, None)
        assert isinstance(candidates, list)
        assert len(candidates) > 0
        assert all(isinstance(algo, str) for algo in candidates)

        # Test with constraints
        constraints = OptimizationConstraintsDTO(
            max_memory_mb=100, max_training_time_seconds=30
        )

        constrained_candidates = self.service._filter_algorithm_candidates(
            characteristics, constraints
        )
        assert isinstance(constrained_candidates, list)
        # Should have fewer or equal candidates due to constraints
        assert len(constrained_candidates) <= len(candidates)

    @pytest.mark.asyncio
    async def test_recommend_algorithm_basic(self):
        """Test basic algorithm recommendation."""
        recommendation = await self.service.recommend_algorithm(self.test_dataset)

        assert isinstance(recommendation, SelectionRecommendationDTO)
        assert len(recommendation.recommended_algorithms) > 0
        assert len(recommendation.confidence_scores) > 0
        assert isinstance(recommendation.reasoning, list)
        assert isinstance(
            recommendation.dataset_characteristics, DatasetCharacteristicsDTO
        )
        assert isinstance(recommendation.timestamp, datetime)

        # Check that confidence scores are valid
        for algo, score in recommendation.confidence_scores.items():
            assert isinstance(algo, str)
            assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_recommend_algorithm_with_constraints(self):
        """Test algorithm recommendation with constraints."""
        constraints = OptimizationConstraintsDTO(
            max_training_time_seconds=60,
            max_memory_mb=500,
            min_accuracy=0.7,
            require_interpretability=True,
        )

        recommendation = await self.service.recommend_algorithm(
            self.test_dataset, constraints=constraints
        )

        assert isinstance(recommendation, SelectionRecommendationDTO)
        assert len(recommendation.recommended_algorithms) > 0

        # Check that recommendation context includes constraint information
        assert "total_candidates" in recommendation.selection_context

    @pytest.mark.asyncio
    async def test_learn_from_result(self):
        """Test learning from algorithm selection result."""
        performance = AlgorithmPerformanceDTO(
            primary_metric=0.85,
            training_time_seconds=45.0,
            memory_usage_mb=200.0,
            secondary_metrics={"precision": 0.87, "recall": 0.83},
        )

        initial_history_size = len(self.service.selection_history)

        await self.service.learn_from_result(
            dataset=self.test_dataset,
            algorithm="isolation_forest",
            performance=performance,
            selection_context={"test": True},
        )

        # Check that history was updated
        assert len(self.service.selection_history) == initial_history_size + 1

        # Check the added entry
        latest_entry = self.service.selection_history[-1]
        assert latest_entry.selected_algorithm == "isolation_forest"
        assert latest_entry.performance.primary_metric == 0.85
        assert latest_entry.selection_context["test"] is True

    @pytest.mark.asyncio
    async def test_benchmark_algorithms(self):
        """Test algorithm benchmarking."""
        # Mock the single algorithm benchmark method
        with patch.object(
            self.service, "_benchmark_single_algorithm"
        ) as mock_benchmark:
            mock_benchmark.return_value = Mock(
                algorithm_name="test_algo",
                mean_score=0.8,
                std_score=0.05,
                cv_scores=[0.75, 0.8, 0.85],
                training_time_seconds=30.0,
                memory_usage_mb=150.0,
                hyperparameters={},
                additional_metrics={},
            )

            benchmarks = await self.service.benchmark_algorithms(
                dataset=self.test_dataset, algorithms=["test_algo"], cv_folds=3
            )

            assert len(benchmarks) == 1
            assert benchmarks[0].algorithm_name == "test_algo"
            assert benchmarks[0].mean_score == 0.8
            assert mock_benchmark.called

    @pytest.mark.asyncio
    async def test_get_learning_insights(self):
        """Test learning insights generation."""
        # Add some dummy history
        performance1 = AlgorithmPerformanceDTO(
            primary_metric=0.8, training_time_seconds=30, memory_usage_mb=100
        )
        performance2 = AlgorithmPerformanceDTO(
            primary_metric=0.75, training_time_seconds=45, memory_usage_mb=150
        )

        await self.service.learn_from_result(
            self.test_dataset, "algo1", performance1, {}
        )
        await self.service.learn_from_result(
            self.test_dataset, "algo2", performance2, {}
        )

        insights = await self.service.get_learning_insights(min_samples=1)

        assert insights.total_selections >= 2
        assert isinstance(insights.algorithm_performance_stats, dict)
        assert isinstance(insights.dataset_type_preferences, dict)
        assert isinstance(insights.performance_trends, dict)
        assert isinstance(insights.generated_at, datetime)
        assert 0.0 <= insights.recommendation_confidence <= 1.0

    def test_get_service_info(self):
        """Test service information retrieval."""
        info = self.service.get_service_info()

        assert isinstance(info, dict)
        assert "meta_learning_enabled" in info
        assert "performance_prediction_enabled" in info
        assert "historical_learning_enabled" in info
        assert "selection_history_size" in info
        assert "available_algorithms" in info
        assert "algorithm_count" in info

        assert info["meta_learning_enabled"] is True
        assert info["performance_prediction_enabled"] is True
        assert info["historical_learning_enabled"] is True
        assert isinstance(info["available_algorithms"], list)
        assert info["algorithm_count"] > 0

    def test_characteristics_to_features(self):
        """Test dataset characteristics to feature vector conversion."""
        characteristics = DatasetCharacteristicsDTO(
            n_samples=1000,
            n_features=4,
            n_numeric_features=3,
            n_categorical_features=1,
            feature_density=0.9,
            missing_value_ratio=0.0,
            outlier_ratio=0.05,
            mean_feature_correlation=0.1,
            feature_variance_ratio=0.2,
            data_dimensionality_ratio=0.004,
            skewness_mean=0.1,
            kurtosis_mean=0.1,
            class_imbalance=0.0,
        )

        features = self.service._characteristics_to_features(characteristics)

        assert isinstance(features, list)
        assert len(features) == 13  # Number of characteristics
        assert all(isinstance(f, float) for f in features)

    def test_compute_dataset_hash(self):
        """Test dataset hash computation."""
        hash1 = self.service._compute_dataset_hash(self.test_dataset)
        hash2 = self.service._compute_dataset_hash(self.test_dataset)

        assert isinstance(hash1, str)
        assert hash1 == hash2  # Same dataset should produce same hash

        # Different dataset should produce different hash
        different_data = pd.DataFrame({"col1": [1, 2, 3]})
        different_dataset = Dataset(
            name="different", data=different_data, features=["col1"]
        )
        hash3 = self.service._compute_dataset_hash(different_dataset)

        assert hash3 != hash1

    @pytest.mark.asyncio
    async def test_rule_based_recommendation(self):
        """Test rule-based recommendation generation."""
        characteristics = DatasetCharacteristicsDTO(
            n_samples=500,  # Small dataset
            n_features=4,
            n_numeric_features=3,
            n_categorical_features=1,
            feature_density=0.9,
            missing_value_ratio=0.0,
            outlier_ratio=0.15,  # High outlier ratio
            mean_feature_correlation=0.1,
            feature_variance_ratio=0.2,
            data_dimensionality_ratio=0.008,
            skewness_mean=0.1,
            kurtosis_mean=0.1,
            class_imbalance=0.0,
        )

        candidates = ["isolation_forest", "local_outlier_factor", "one_class_svm"]

        scores = await self.service._rule_based_recommendation(
            characteristics, candidates
        )

        assert isinstance(scores, dict)
        assert len(scores) == len(candidates)
        assert all(0.0 <= score <= 1.0 for score in scores.values())

        # For small dataset with high outlier ratio, isolation_forest should score well
        assert "isolation_forest" in scores

    @pytest.mark.asyncio
    async def test_historical_similarity_recommendation(self):
        """Test historical similarity-based recommendation."""
        characteristics = DatasetCharacteristicsDTO(
            n_samples=1000,
            n_features=4,
            n_numeric_features=3,
            n_categorical_features=1,
            feature_density=0.9,
            missing_value_ratio=0.0,
            outlier_ratio=0.05,
            mean_feature_correlation=0.1,
            feature_variance_ratio=0.2,
            data_dimensionality_ratio=0.004,
            skewness_mean=0.1,
            kurtosis_mean=0.1,
            class_imbalance=0.0,
        )

        candidates = ["isolation_forest", "local_outlier_factor"]

        # Test with empty history
        scores = await self.service._historical_similarity_recommendation(
            characteristics, candidates
        )
        assert isinstance(scores, dict)
        assert len(scores) == len(candidates)

        # Add some history and test again
        performance = AlgorithmPerformanceDTO(
            primary_metric=0.8, training_time_seconds=30, memory_usage_mb=100
        )
        await self.service.learn_from_result(
            self.test_dataset, "isolation_forest", performance, {}
        )

        scores_with_history = await self.service._historical_similarity_recommendation(
            characteristics, candidates
        )
        assert isinstance(scores_with_history, dict)
        assert len(scores_with_history) == len(candidates)

    def test_analyze_algorithm_performance(self):
        """Test algorithm performance analysis."""
        # Add some dummy history manually
        from pynomaly.application.dto.selection_dto import SelectionHistoryDTO

        characteristics = DatasetCharacteristicsDTO(
            n_samples=1000,
            n_features=4,
            n_numeric_features=3,
            n_categorical_features=1,
            feature_density=0.9,
            missing_value_ratio=0.0,
            outlier_ratio=0.05,
            mean_feature_correlation=0.1,
            feature_variance_ratio=0.2,
            data_dimensionality_ratio=0.004,
            skewness_mean=0.1,
            kurtosis_mean=0.1,
            class_imbalance=0.0,
        )

        # Add some history entries
        for i, algo in enumerate(["algo1", "algo2", "algo1"]):
            performance = AlgorithmPerformanceDTO(
                primary_metric=0.8 + i * 0.05,
                training_time_seconds=30,
                memory_usage_mb=100,
            )

            history_entry = SelectionHistoryDTO(
                dataset_characteristics=characteristics,
                selected_algorithm=algo,
                performance=performance,
                selection_context={},
                timestamp=datetime.now(),
                dataset_hash="test_hash",
            )

            self.service.selection_history.append(history_entry)

        stats = self.service._analyze_algorithm_performance()

        assert isinstance(stats, dict)
        assert "algo1" in stats
        assert "algo2" in stats

        # Check statistics structure
        for algo_stats in stats.values():
            assert "mean" in algo_stats
            assert "std" in algo_stats
            assert "count" in algo_stats
            assert isinstance(algo_stats["mean"], float)
            assert isinstance(algo_stats["count"], int)

    def test_analyze_dataset_preferences(self):
        """Test dataset type preferences analysis."""
        preferences = self.service._analyze_dataset_preferences()

        assert isinstance(preferences, dict)
        assert "small_datasets" in preferences
        assert "large_datasets" in preferences
        assert "high_dimensional" in preferences
        assert "sparse_data" in preferences

        # All values should be lists
        for category_prefs in preferences.values():
            assert isinstance(category_prefs, list)

    def test_calculate_recommendation_confidence(self):
        """Test recommendation confidence calculation."""
        confidence = self.service._calculate_recommendation_confidence()

        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    def test_initialize_algorithm_registry(self):
        """Test algorithm registry initialization."""
        registry = self.service._initialize_algorithm_registry()

        assert isinstance(registry, dict)
        assert len(registry) > 0

        # Check registry structure
        for algo_name, algo_info in registry.items():
            assert isinstance(algo_name, str)
            assert isinstance(algo_info, dict)
            assert "min_samples" in algo_info
            assert "max_features" in algo_info
            assert "memory_usage" in algo_info
            assert "training_time" in algo_info

            # Check data types
            assert isinstance(algo_info["min_samples"], int)
            assert isinstance(algo_info["max_features"], int)
            assert isinstance(algo_info["memory_usage"], int | float)
            assert isinstance(algo_info["training_time"], int | float)

    @pytest.mark.asyncio
    async def test_meta_learning_recommendation_no_model(self):
        """Test meta-learning recommendation when no model is available."""
        # Ensure no meta-learner is available
        self.service.meta_learner = None

        characteristics = DatasetCharacteristicsDTO(
            n_samples=1000,
            n_features=4,
            n_numeric_features=3,
            n_categorical_features=1,
            feature_density=0.9,
            missing_value_ratio=0.0,
            outlier_ratio=0.05,
            mean_feature_correlation=0.1,
            feature_variance_ratio=0.2,
            data_dimensionality_ratio=0.004,
            skewness_mean=0.1,
            kurtosis_mean=0.1,
            class_imbalance=0.0,
        )

        candidates = ["isolation_forest", "local_outlier_factor"]

        scores = await self.service._meta_learning_recommendation(
            characteristics, candidates, None
        )

        assert isinstance(scores, dict)
        assert len(scores) == len(candidates)
        assert all(score == 0.0 for score in scores.values())

    @pytest.mark.asyncio
    async def test_predict_performance_no_predictor(self):
        """Test performance prediction when no predictor is available."""
        # Ensure no performance predictor is available
        self.service.performance_predictor = None

        recommendation = SelectionRecommendationDTO(
            recommended_algorithms=["isolation_forest"],
            confidence_scores={"isolation_forest": 0.8},
            reasoning=["Test reasoning"],
            dataset_characteristics=DatasetCharacteristicsDTO(
                n_samples=1000,
                n_features=4,
                n_numeric_features=3,
                n_categorical_features=1,
                feature_density=0.9,
                missing_value_ratio=0.0,
                outlier_ratio=0.05,
                mean_feature_correlation=0.1,
                feature_variance_ratio=0.2,
                data_dimensionality_ratio=0.004,
                skewness_mean=0.1,
                kurtosis_mean=0.1,
                class_imbalance=0.0,
            ),
            selection_context={},
            timestamp=datetime.now(),
        )

        characteristics = recommendation.dataset_characteristics

        result = await self.service._predict_performance(
            recommendation, characteristics
        )

        # Should return the same recommendation unchanged
        assert result == recommendation

    def test_analyze_performance_trends(self):
        """Test performance trends analysis."""
        trends = self.service._analyze_performance_trends()

        assert isinstance(trends, dict)
        assert "overall_performance" in trends
        assert isinstance(trends["overall_performance"], list)

    @pytest.mark.asyncio
    async def test_analyze_feature_importance_no_model(self):
        """Test feature importance analysis when no meta-learner is available."""
        # Ensure no meta-learner is available
        self.service.meta_learner = None

        importance = await self.service._analyze_feature_importance()

        assert isinstance(importance, dict)
        assert len(importance) == 0  # Should be empty when no model is available

    @pytest.mark.asyncio
    async def test_get_meta_model_accuracy_insufficient_data(self):
        """Test meta-model accuracy when insufficient data is available."""
        # Ensure minimal history
        self.service.selection_history = []

        accuracy = await self.service._get_meta_model_accuracy()

        assert accuracy is None  # Should return None for insufficient data


class TestDatasetCharacteristicsDTO:
    """Test dataset characteristics DTO."""

    def test_valid_characteristics(self):
        """Test valid characteristics creation."""
        characteristics = DatasetCharacteristicsDTO(
            n_samples=1000,
            n_features=10,
            n_numeric_features=8,
            n_categorical_features=2,
            feature_density=0.95,
            missing_value_ratio=0.02,
            outlier_ratio=0.05,
            mean_feature_correlation=0.15,
            feature_variance_ratio=0.3,
            data_dimensionality_ratio=0.01,
            skewness_mean=0.1,
            kurtosis_mean=0.2,
            class_imbalance=0.1,
        )

        assert characteristics.n_samples == 1000
        assert characteristics.n_features == 10
        assert characteristics.feature_density == 0.95
        assert characteristics.outlier_ratio == 0.05

    def test_characteristics_validation(self):
        """Test characteristics validation."""
        # Test invalid feature_density (> 1)
        with pytest.raises(ValueError):
            DatasetCharacteristicsDTO(
                n_samples=1000,
                n_features=10,
                n_numeric_features=8,
                n_categorical_features=2,
                feature_density=1.5,  # Invalid: > 1
                missing_value_ratio=0.02,
                outlier_ratio=0.05,
                mean_feature_correlation=0.15,
                feature_variance_ratio=0.3,
                data_dimensionality_ratio=0.01,
                skewness_mean=0.1,
                kurtosis_mean=0.2,
                class_imbalance=0.1,
            )

        # Test negative n_samples
        with pytest.raises(ValueError):
            DatasetCharacteristicsDTO(
                n_samples=-100,  # Invalid: < 0
                n_features=10,
                n_numeric_features=8,
                n_categorical_features=2,
                feature_density=0.95,
                missing_value_ratio=0.02,
                outlier_ratio=0.05,
                mean_feature_correlation=0.15,
                feature_variance_ratio=0.3,
                data_dimensionality_ratio=0.01,
                skewness_mean=0.1,
                kurtosis_mean=0.2,
                class_imbalance=0.1,
            )


class TestOptimizationConstraintsDTO:
    """Test optimization constraints DTO."""

    def test_valid_constraints(self):
        """Test valid constraints creation."""
        constraints = OptimizationConstraintsDTO(
            max_training_time_seconds=120.0,
            max_memory_mb=1000.0,
            min_accuracy=0.8,
            require_interpretability=True,
            gpu_available=True,
        )

        assert constraints.max_training_time_seconds == 120.0
        assert constraints.max_memory_mb == 1000.0
        assert constraints.min_accuracy == 0.8
        assert constraints.require_interpretability is True
        assert constraints.gpu_available is True

    def test_constraints_validation(self):
        """Test constraints validation."""
        # Test negative training time
        with pytest.raises(ValueError):
            OptimizationConstraintsDTO(max_training_time_seconds=-10.0)

        # Test accuracy > 1
        with pytest.raises(ValueError):
            OptimizationConstraintsDTO(min_accuracy=1.5)


class TestAlgorithmPerformanceDTO:
    """Test algorithm performance DTO."""

    def test_valid_performance(self):
        """Test valid performance creation."""
        performance = AlgorithmPerformanceDTO(
            primary_metric=0.85,
            training_time_seconds=45.0,
            memory_usage_mb=200.0,
            secondary_metrics={"precision": 0.87, "recall": 0.83},
            stability_score=0.9,
            interpretability_score=0.7,
            confidence_score=0.8,
        )

        assert performance.primary_metric == 0.85
        assert performance.training_time_seconds == 45.0
        assert performance.secondary_metrics["precision"] == 0.87
        assert performance.stability_score == 0.9

    def test_performance_validation(self):
        """Test performance validation."""
        # Test primary_metric > 1
        with pytest.raises(ValueError):
            AlgorithmPerformanceDTO(
                primary_metric=1.5,  # Invalid: > 1
                training_time_seconds=45.0,
                memory_usage_mb=200.0,
            )

        # Test negative training time
        with pytest.raises(ValueError):
            AlgorithmPerformanceDTO(
                primary_metric=0.85,
                training_time_seconds=-10.0,  # Invalid: < 0
                memory_usage_mb=200.0,
            )
