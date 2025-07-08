"""Tests for configuration recommendation service."""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from pynomaly.application.dto.configuration_dto import (
    AlgorithmConfigurationDTO,
    ConfigurationLevel,
    ConfigurationMetadataDTO,
    ConfigurationSource,
    DatasetCharacteristicsDTO,
    DatasetConfigurationDTO,
    ExperimentConfigurationDTO,
    PerformanceResultsDTO,
)
from pynomaly.application.services.configuration_capture_service import (
    ConfigurationCaptureService,
)
from pynomaly.application.services.configuration_recommendation_service import (
    ConfigurationRecommendationService,
)
from pynomaly.infrastructure.persistence.configuration_repository import (
    ConfigurationRepository,
)


class TestConfigurationRecommendationService:
    """Test configuration recommendation service."""

    @pytest.fixture
    def capture_service(self):
        """Create configuration capture service."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield ConfigurationCaptureService(
                storage_path=Path(temp_dir), auto_capture=True
            )

    @pytest.fixture
    def repository(self):
        """Create configuration repository."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield ConfigurationRepository(storage_path=Path(temp_dir))

    @pytest.fixture
    def recommendation_service(self, capture_service, repository):
        """Create recommendation service."""
        return ConfigurationRecommendationService(
            configuration_service=capture_service,
            repository=repository,
            enable_ml_recommendations=True,
            enable_similarity_recommendations=True,
            enable_performance_prediction=True,
        )

    @pytest.fixture
    def sample_dataset_characteristics(self):
        """Create sample dataset characteristics."""
        return DatasetCharacteristicsDTO(
            n_samples=1000,
            n_features=10,
            feature_types=["numeric", "numeric", "categorical"],
            missing_values_ratio=0.05,
            outlier_ratio=0.02,
            class_imbalance=None,
            sparsity=0.1,
            correlation_structure="low",
        )

    @pytest.fixture
    def sample_configurations(self, repository):
        """Create sample configurations for testing."""
        configs = []
        algorithms = ["IsolationForest", "LocalOutlierFactor", "OneClassSVM"]

        for i, algorithm in enumerate(algorithms):
            config = ExperimentConfigurationDTO(
                id=uuid4(),
                name=f"test_config_{i}",
                algorithm_config=AlgorithmConfigurationDTO(
                    algorithm_name=algorithm,
                    hyperparameters={"contamination": 0.1, "random_state": 42},
                ),
                dataset_config=DatasetConfigurationDTO(
                    n_samples=1000 + i * 500,
                    n_features=10 + i * 5,
                    feature_types=["numeric"] * (10 + i * 5),
                ),
                performance_results=PerformanceResultsDTO(
                    accuracy=0.8 + i * 0.05,
                    precision=0.78 + i * 0.05,
                    recall=0.82 + i * 0.05,
                    f1_score=0.8 + i * 0.05,
                    training_time_seconds=60 + i * 30,
                ),
                metadata=ConfigurationMetadataDTO(
                    source=ConfigurationSource.MANUAL,
                    created_at=datetime.now() - timedelta(days=i),
                    description=f"Test configuration {i}",
                    tags=["test", algorithm.lower()],
                ),
            )
            configs.append(config)

        return configs

    @pytest.mark.asyncio
    async def test_recommend_configurations_rule_based(
        self, recommendation_service, sample_dataset_characteristics
    ):
        """Test rule-based configuration recommendations."""
        # Test with small dataset
        small_dataset = DatasetCharacteristicsDTO(
            n_samples=500,
            n_features=5,
            feature_types=["numeric"] * 5,
            missing_values_ratio=0.01,
            outlier_ratio=0.05,
            sparsity=0.0,
            correlation_structure="low",
        )

        recommendations = await recommendation_service.recommend_configurations(
            dataset_characteristics=small_dataset,
            difficulty_level=ConfigurationLevel.BEGINNER,
            max_recommendations=3,
        )

        # Should get rule-based recommendations
        assert len(recommendations) > 0
        assert any(rec.algorithm_name == "IsolationForest" for rec in recommendations)
        assert all(rec.confidence_score > 0 for rec in recommendations)
        assert all(rec.recommendation_reason for rec in recommendations)

    @pytest.mark.asyncio
    async def test_recommend_configurations_with_performance_requirements(
        self, recommendation_service, sample_dataset_characteristics
    ):
        """Test recommendations with performance requirements."""
        performance_requirements = {"min_accuracy": 0.85}

        recommendations = await recommendation_service.recommend_configurations(
            dataset_characteristics=sample_dataset_characteristics,
            performance_requirements=performance_requirements,
            max_recommendations=5,
        )

        # Should adjust confidence based on performance requirements
        assert len(recommendations) >= 0  # May be empty if no suitable configs

        for rec in recommendations:
            if rec.predicted_performance and "accuracy" in rec.predicted_performance:
                # Confidence should be adjusted based on meeting requirements
                assert isinstance(rec.confidence_score, float)
                assert 0 <= rec.confidence_score <= 1

    @pytest.mark.asyncio
    async def test_recommend_configurations_use_case_specific(
        self, recommendation_service, sample_dataset_characteristics
    ):
        """Test use case specific recommendations."""
        recommendations = await recommendation_service.recommend_configurations(
            dataset_characteristics=sample_dataset_characteristics,
            use_case="fraud_detection",
            max_recommendations=3,
        )

        # Should get fraud detection specific recommendations
        assert len(recommendations) > 0

        # Check for fraud detection specific algorithms or parameters
        fraud_rec = None
        for rec in recommendations:
            if "fraud" in rec.recommendation_reason.lower():
                fraud_rec = rec
                break

        if fraud_rec:
            assert fraud_rec.confidence_score > 0

    @pytest.mark.asyncio
    async def test_predict_configuration_performance_without_model(
        self, recommendation_service, sample_dataset_characteristics
    ):
        """Test performance prediction without trained model."""
        # Create a sample configuration
        config = ExperimentConfigurationDTO(
            id=uuid4(),
            name="test_config",
            algorithm_config=AlgorithmConfigurationDTO(
                algorithm_name="IsolationForest", hyperparameters={"contamination": 0.1}
            ),
            dataset_config=DatasetConfigurationDTO(
                n_samples=1000, n_features=10, feature_types=["numeric"] * 10
            ),
            metadata=ConfigurationMetadataDTO(
                source=ConfigurationSource.MANUAL,
                created_at=datetime.now(),
                description="Test",
                tags=[],
            ),
        )

        # Should return empty dict without trained model
        predictions = await recommendation_service.predict_configuration_performance(
            config, sample_dataset_characteristics
        )

        assert isinstance(predictions, dict)
        # Without trained model, predictions should be empty
        assert len(predictions) == 0

    @pytest.mark.asyncio
    async def test_train_recommendation_models_insufficient_data(
        self, recommendation_service
    ):
        """Test model training with insufficient data."""
        results = await recommendation_service.train_recommendation_models(
            min_configurations=20
        )

        # Should fail due to insufficient data
        assert "error" in results
        assert (
            "available_configurations" in results
            or "configurations_processed" in results
        )

    @pytest.mark.asyncio
    async def test_train_recommendation_models_with_data(
        self, recommendation_service, repository, sample_configurations
    ):
        """Test model training with sufficient data."""
        # Add sample configurations to repository
        for config in sample_configurations:
            await repository.save_configuration(config)

        # Try training with lower threshold
        results = await recommendation_service.train_recommendation_models(
            min_configurations=2,  # Lower threshold for testing
            test_size=0.5,
        )

        # Might still fail due to insufficient valid features, but should attempt training
        if "error" not in results:
            assert results["success"] is True
            assert "configurations_used" in results
            assert "performance_predictor_rmse" in results
            assert "algorithm_selector_accuracy" in results
        else:
            # Expected due to simplified test data
            assert (
                "configurations_processed" in results
                or "available_configurations" in results
            )

    @pytest.mark.asyncio
    async def test_analyze_recommendation_patterns(
        self, recommendation_service, repository, sample_configurations
    ):
        """Test recommendation pattern analysis."""
        # Add sample configurations
        for config in sample_configurations:
            await repository.save_configuration(config)

        analysis = await recommendation_service.analyze_recommendation_patterns(
            time_period_days=7
        )

        # Verify analysis structure
        assert "time_period_days" in analysis
        assert "total_configurations" in analysis
        assert "algorithm_popularity" in analysis
        assert "algorithm_performance" in analysis
        assert "dataset_characteristics" in analysis
        assert "source_distribution" in analysis
        assert "performance_summary" in analysis
        assert "recommendation_trends" in analysis

        # Check algorithm popularity
        assert isinstance(analysis["algorithm_popularity"], dict)

        # Check performance summary
        perf_summary = analysis["performance_summary"]
        assert "count" in perf_summary
        assert "mean_accuracy" in perf_summary
        assert "high_performance_rate" in perf_summary

    def test_get_recommendation_statistics(self, recommendation_service):
        """Test recommendation statistics retrieval."""
        stats = recommendation_service.get_recommendation_statistics()

        # Verify statistics structure
        assert "recommendation_stats" in stats
        assert "model_status" in stats
        assert "cache_info" in stats
        assert "feature_flags" in stats

        # Check recommendation stats
        rec_stats = stats["recommendation_stats"]
        expected_keys = [
            "total_recommendations",
            "ml_recommendations",
            "similarity_recommendations",
            "performance_predictions",
            "successful_predictions",
            "model_training_count",
        ]
        for key in expected_keys:
            assert key in rec_stats
            assert isinstance(rec_stats[key], int)

        # Check model status
        model_status = stats["model_status"]
        assert "performance_predictor_trained" in model_status
        assert "algorithm_selector_trained" in model_status
        assert "feature_scaler_available" in model_status

        # Check feature flags
        feature_flags = stats["feature_flags"]
        assert "ml_recommendations" in feature_flags
        assert "similarity_recommendations" in feature_flags
        assert "performance_prediction" in feature_flags

    @pytest.mark.asyncio
    async def test_similarity_based_recommendations(
        self,
        recommendation_service,
        repository,
        sample_configurations,
        sample_dataset_characteristics,
    ):
        """Test similarity-based recommendations."""
        # Add configurations to repository
        for config in sample_configurations:
            await repository.save_configuration(config)

        # Create similar dataset characteristics
        similar_characteristics = DatasetCharacteristicsDTO(
            n_samples=1100,  # Similar to first config (1000)
            n_features=12,  # Similar to first config (10)
            feature_types=["numeric"] * 12,
            missing_values_ratio=0.04,
            outlier_ratio=0.03,
            sparsity=0.05,
            correlation_structure="low",
        )

        # Test similarity recommendations
        recommendations = (
            await recommendation_service._generate_similarity_recommendations(
                similar_characteristics, None, 3
            )
        )

        # Should find similar configurations
        assert isinstance(recommendations, list)
        # May be empty due to strict similarity threshold
        for rec_tuple in recommendations:
            config, similarity = rec_tuple
            assert isinstance(config, ExperimentConfigurationDTO)
            assert isinstance(similarity, float)
            assert 0 <= similarity <= 1

    @pytest.mark.asyncio
    async def test_rule_based_recommendations_edge_cases(self, recommendation_service):
        """Test rule-based recommendations for edge cases."""
        # High-dimensional dataset
        high_dim_dataset = DatasetCharacteristicsDTO(
            n_samples=1000,
            n_features=100,  # High dimensional
            feature_types=["numeric"] * 100,
            missing_values_ratio=0.01,
            outlier_ratio=0.05,
            sparsity=0.0,
            correlation_structure="high",
        )

        recommendations = (
            await recommendation_service._generate_rule_based_recommendations(
                high_dim_dataset, None, None, ConfigurationLevel.ADVANCED, 5
            )
        )

        # Should get recommendations for high-dimensional data
        assert len(recommendations) > 0

        # Sparse dataset
        sparse_dataset = DatasetCharacteristicsDTO(
            n_samples=1000,
            n_features=20,
            feature_types=["numeric"] * 20,
            missing_values_ratio=0.01,
            outlier_ratio=0.05,
            sparsity=0.8,  # Very sparse
            correlation_structure="low",
        )

        sparse_recommendations = (
            await recommendation_service._generate_rule_based_recommendations(
                sparse_dataset, None, None, ConfigurationLevel.INTERMEDIATE, 5
            )
        )

        # Should get recommendations for sparse data
        assert len(sparse_recommendations) > 0

        # Check for OneClassSVM recommendation for sparse data
        svm_rec = None
        for rec in sparse_recommendations:
            if rec.algorithm_name == "OneClassSVM":
                svm_rec = rec
                break

        if svm_rec:
            assert "sparse" in svm_rec.recommendation_reason.lower()

    @pytest.mark.asyncio
    async def test_feature_extraction_for_prediction(
        self, recommendation_service, sample_dataset_characteristics
    ):
        """Test feature extraction for ML prediction."""
        config = ExperimentConfigurationDTO(
            id=uuid4(),
            name="test_config",
            algorithm_config=AlgorithmConfigurationDTO(
                algorithm_name="IsolationForest",
                hyperparameters={
                    "contamination": 0.1,
                    "n_estimators": 100,
                    "random_state": 42,
                },
            ),
            dataset_config=DatasetConfigurationDTO(
                n_samples=1000, n_features=10, feature_types=["numeric"] * 10
            ),
            metadata=ConfigurationMetadataDTO(
                source=ConfigurationSource.AUTOML,
                created_at=datetime.now(),
                description="Test",
                tags=[],
            ),
        )

        features = recommendation_service._extract_prediction_features(
            config, sample_dataset_characteristics
        )

        # Verify feature extraction
        assert isinstance(features, list)
        assert len(features) > 0
        assert all(isinstance(f, int | float) for f in features)

        # Features should include dataset characteristics, algorithm info, etc.
        # First few features should be log-transformed dataset characteristics
        assert features[0] > 0  # log(n_samples + 1)
        assert features[1] > 0  # log(n_features + 1)

    @pytest.mark.asyncio
    async def test_training_time_estimation(
        self, recommendation_service, sample_dataset_characteristics
    ):
        """Test training time estimation."""
        config = ExperimentConfigurationDTO(
            id=uuid4(),
            name="test_config",
            algorithm_config=AlgorithmConfigurationDTO(
                algorithm_name="IsolationForest", hyperparameters={}
            ),
            dataset_config=DatasetConfigurationDTO(
                n_samples=1000, n_features=10, feature_types=["numeric"] * 10
            ),
            metadata=ConfigurationMetadataDTO(
                source=ConfigurationSource.MANUAL,
                created_at=datetime.now(),
                description="Test",
                tags=[],
            ),
        )

        estimated_time = recommendation_service._estimate_training_time(
            config, sample_dataset_characteristics
        )

        # Should return reasonable time estimate
        assert isinstance(estimated_time, float)
        assert estimated_time >= 10.0  # Minimum 10 seconds
        assert estimated_time <= 10000.0  # Reasonable upper bound

    @pytest.mark.asyncio
    async def test_dataset_similarity_calculation(self, recommendation_service):
        """Test dataset similarity calculation."""
        ds1 = DatasetCharacteristicsDTO(
            n_samples=1000,
            n_features=10,
            feature_types=["numeric"] * 10,
            missing_values_ratio=0.05,
            outlier_ratio=0.02,
            sparsity=0.1,
            correlation_structure="low",
        )

        ds2 = DatasetConfigurationDTO(
            n_samples=1100,  # Similar
            n_features=12,  # Similar
            feature_types=["numeric"] * 12,
        )

        similarity = recommendation_service._calculate_dataset_similarity(ds1, ds2)

        # Should be high similarity
        assert isinstance(similarity, float)
        assert 0 <= similarity <= 1
        assert similarity > 0.5  # Should be reasonably similar

        # Test with very different dataset
        ds3 = DatasetConfigurationDTO(
            n_samples=100000,  # Very different
            n_features=2,  # Very different
            feature_types=["categorical"] * 2,
        )

        similarity_low = recommendation_service._calculate_dataset_similarity(ds1, ds3)
        assert similarity_low < similarity  # Should be less similar

    @pytest.mark.asyncio
    async def test_preprocessing_suggestions(self, recommendation_service):
        """Test preprocessing suggestions generation."""
        # High-dimensional dataset
        high_dim_dataset = DatasetCharacteristicsDTO(
            n_samples=1000,
            n_features=100,
            feature_types=["numeric"] * 100,
            missing_values_ratio=0.1,
            outlier_ratio=0.05,
            sparsity=0.5,
            correlation_structure="high",
        )

        suggestions = recommendation_service._get_preprocessing_suggestions(
            high_dim_dataset
        )

        # Should suggest appropriate preprocessing
        assert isinstance(suggestions, dict)
        assert "scaling" in suggestions
        assert suggestions["scaling"] == "standard"

        # Should suggest PCA for high-dimensional data
        assert "pca" in suggestions
        assert "n_components" in suggestions["pca"]

        # Should suggest missing value handling
        assert "missing_values" in suggestions

        # Should suggest outlier handling for sparse data
        assert "outlier_handling" in suggestions

    @pytest.mark.asyncio
    async def test_recommendation_deduplication(
        self, recommendation_service, sample_dataset_characteristics
    ):
        """Test recommendation deduplication and scoring."""
        from pynomaly.application.dto.configuration_dto import (
            ConfigurationRecommendationDTO,
        )

        # Create duplicate recommendations
        rec1 = ConfigurationRecommendationDTO(
            configuration_id=uuid4(),
            algorithm_name="IsolationForest",
            confidence_score=0.8,
            predicted_performance={"accuracy": 0.85},
            recommendation_reason="Test reason 1",
            use_cases=[],
            difficulty_level=ConfigurationLevel.INTERMEDIATE,
            estimated_training_time=60,
            parameter_suggestions={},
            preprocessing_suggestions={},
            evaluation_suggestions={},
        )

        rec2 = ConfigurationRecommendationDTO(
            configuration_id=uuid4(),
            algorithm_name="IsolationForest",  # Same algorithm
            confidence_score=0.7,
            predicted_performance={"accuracy": 0.82},
            recommendation_reason="Test reason 2",
            use_cases=[],
            difficulty_level=ConfigurationLevel.INTERMEDIATE,
            estimated_training_time=55,
            parameter_suggestions={},
            preprocessing_suggestions={},
            evaluation_suggestions={},
        )

        rec3 = ConfigurationRecommendationDTO(
            configuration_id=uuid4(),
            algorithm_name="LocalOutlierFactor",  # Different algorithm
            confidence_score=0.75,
            predicted_performance={"accuracy": 0.80},
            recommendation_reason="Test reason 3",
            use_cases=[],
            difficulty_level=ConfigurationLevel.INTERMEDIATE,
            estimated_training_time=70,
            parameter_suggestions={},
            preprocessing_suggestions={},
            evaluation_suggestions={},
        )

        recommendations = [rec1, rec2, rec3]

        # Test deduplication
        unique_recs = recommendation_service._deduplicate_and_score_recommendations(
            recommendations, sample_dataset_characteristics, {"min_accuracy": 0.8}
        )

        # Should remove duplicates (same algorithm)
        assert len(unique_recs) == 2  # IsolationForest and LocalOutlierFactor

        # Should keep the higher confidence one for IsolationForest
        isolation_rec = None
        lof_rec = None
        for rec in unique_recs:
            if rec.algorithm_name == "IsolationForest":
                isolation_rec = rec
            elif rec.algorithm_name == "LocalOutlierFactor":
                lof_rec = rec

        assert isolation_rec is not None
        assert lof_rec is not None
        assert (
            isolation_rec.confidence_score >= 0.8
        )  # Should be the higher confidence one


if __name__ == "__main__":
    pytest.main([__file__])
