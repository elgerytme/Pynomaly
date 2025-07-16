"""Tests for extracted services following single responsibility principle."""

import asyncio

import numpy as np
import pandas as pd
import pytest

from src.monorepo.application.services.data_profiling_service import (
    DataProfilingService,
)
from src.monorepo.application.services.data_validation_service import (
    DataValidationService,
)
from src.monorepo.application.services.feature_engineering_service import (
    FeatureEngineeringService,
)
from src.monorepo.application.services.interfaces.pipeline_services import (
    DataProfile,
    DataValidationResult,
    FeatureEngineeringResult,
)


class TestDataValidationService:
    """Test suite for DataValidationService."""

    @pytest.fixture
    def validation_service(self):
        """Create a validation service instance."""
        return DataValidationService(min_quality_threshold=0.7)

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [10.5, 20.1, 30.7, 40.2, 50.8],
                "feature3": ["A", "B", "C", "D", "E"],
                "feature4": [1, 1, 1, 1, 1],  # Constant feature
            }
        )

    @pytest.fixture
    def sample_target(self):
        """Create sample target data."""
        return pd.Series([0, 1, 0, 1, 0], name="target")

    @pytest.mark.asyncio
    async def test_validate_data_basic(
        self, validation_service, sample_data, sample_target
    ):
        """Test basic data validation."""
        result = await validation_service.validate_data(sample_data, sample_target)

        assert isinstance(result, DataValidationResult)
        assert isinstance(result.is_valid, bool)
        assert isinstance(result.quality_score, float)
        assert 0 <= result.quality_score <= 1
        assert isinstance(result.statistics, dict)
        assert isinstance(result.issues, list)
        assert isinstance(result.recommendations, list)

    @pytest.mark.asyncio
    async def test_validate_data_without_target(self, validation_service, sample_data):
        """Test data validation without target variable."""
        result = await validation_service.validate_data(sample_data)

        assert isinstance(result, DataValidationResult)
        assert "target" not in result.statistics

    @pytest.mark.asyncio
    async def test_validate_data_with_issues(self, validation_service):
        """Test data validation with various issues."""
        # Create problematic data
        problematic_data = pd.DataFrame(
            {
                "feature1": [1, 2, np.nan, np.nan, np.nan],  # High missing values
                "feature2": [1, 1, 1, 1, 1],  # Constant feature
                "feature3": [10, 20, 30, 40, 50],  # Duplicates will be added
            }
        )
        # Add duplicate rows
        problematic_data = pd.concat([problematic_data, problematic_data.iloc[[0, 1]]])

        result = await validation_service.validate_data(problematic_data)

        assert len(result.issues) > 0
        assert len(result.recommendations) > 0
        assert result.quality_score < 0.7

    @pytest.mark.asyncio
    async def test_validate_small_dataset(self, validation_service):
        """Test validation of small dataset."""
        small_data = pd.DataFrame({"feature1": [1, 2]})

        result = await validation_service.validate_data(small_data)

        assert "Small dataset" in str(result.issues) or "Insufficient data" in str(
            result.issues
        )

    @pytest.mark.asyncio
    async def test_validate_empty_dataset(self, validation_service):
        """Test validation of empty dataset."""
        empty_data = pd.DataFrame()

        result = await validation_service.validate_data(empty_data)

        assert not result.is_valid
        assert result.quality_score == 0.0


class TestDataProfilingService:
    """Test suite for DataProfilingService."""

    @pytest.fixture
    def profiling_service(self):
        """Create a profiling service instance."""
        return DataProfilingService(include_advanced_analysis=True)

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame(
            {
                "numeric1": [1, 2, 3, 4, 5],
                "numeric2": [10.5, 20.1, 30.7, 40.2, 50.8],
                "categorical1": ["A", "B", "C", "D", "E"],
                "categorical2": ["X", "Y", "X", "Y", "Z"],
                "boolean1": [True, False, True, False, True],
            }
        )

    @pytest.fixture
    def sample_target(self):
        """Create sample target data."""
        return pd.Series([0, 1, 0, 1, 0], name="target")

    @pytest.mark.asyncio
    async def test_profile_data_basic(
        self, profiling_service, sample_data, sample_target
    ):
        """Test basic data profiling."""
        result = await profiling_service.profile_data(sample_data, sample_target)

        assert isinstance(result, DataProfile)
        assert isinstance(result.basic_stats, dict)
        assert isinstance(result.feature_analysis, dict)
        assert isinstance(result.data_quality, dict)
        assert isinstance(result.sparsity_ratio, float)
        assert isinstance(result.missing_values_ratio, float)
        assert isinstance(result.complexity_score, float)

        # Check basic stats
        assert result.basic_stats["n_samples"] == 5
        assert result.basic_stats["n_features"] == 5
        assert "memory_usage_mb" in result.basic_stats

    @pytest.mark.asyncio
    async def test_profile_data_without_target(self, profiling_service, sample_data):
        """Test data profiling without target variable."""
        result = await profiling_service.profile_data(sample_data)

        assert isinstance(result, DataProfile)
        assert "target_analysis" not in result.basic_stats

    @pytest.mark.asyncio
    async def test_feature_analysis(self, profiling_service, sample_data):
        """Test feature analysis functionality."""
        result = await profiling_service.profile_data(sample_data)

        feature_analysis = result.feature_analysis
        assert "feature_types" in feature_analysis
        assert "numeric_features" in feature_analysis
        assert "categorical_features" in feature_analysis
        assert "boolean_features" in feature_analysis

        # Check feature counts
        assert feature_analysis["numeric_count"] == 2
        assert feature_analysis["categorical_count"] == 2
        assert feature_analysis["boolean_count"] == 1

    @pytest.mark.asyncio
    async def test_data_quality_analysis(self, profiling_service):
        """Test data quality analysis."""
        # Create data with quality issues
        quality_data = pd.DataFrame(
            {
                "feature1": [1, 2, np.nan, 4, 5],  # Missing values
                "feature2": [1, 1, 1, 1, 1],  # Constant feature
                "feature3": [10, 20, 30, 40, 50],
            }
        )
        # Add duplicate rows
        quality_data = pd.concat([quality_data, quality_data.iloc[[0]]])

        result = await profiling_service.profile_data(quality_data)

        data_quality = result.data_quality
        assert "missing_values" in data_quality
        assert "duplicates" in data_quality
        assert "feature_variance" in data_quality

        # Check missing values analysis
        assert data_quality["missing_values"]["total_missing"] > 0
        assert len(data_quality["missing_values"]["columns_with_missing"]) > 0

        # Check duplicates analysis
        assert data_quality["duplicates"]["duplicate_count"] > 0

        # Check constant features
        assert data_quality["feature_variance"]["constant_count"] > 0

    @pytest.mark.asyncio
    async def test_complexity_score_calculation(self, profiling_service):
        """Test complexity score calculation."""
        # Simple data
        simple_data = pd.DataFrame(
            {
                "feature1": [1, 2, 3],
                "feature2": [4, 5, 6],
            }
        )

        result = await profiling_service.profile_data(simple_data)
        simple_complexity = result.complexity_score

        # Complex data
        complex_data = pd.DataFrame(
            {f"feature_{i}": np.random.rand(1000) for i in range(100)}
        )

        result = await profiling_service.profile_data(complex_data)
        complex_complexity = result.complexity_score

        # Complex data should have higher complexity score
        assert complex_complexity > simple_complexity

    @pytest.mark.asyncio
    async def test_profiling_summary(self, profiling_service, sample_data):
        """Test profiling summary generation."""
        result = await profiling_service.profile_data(sample_data)
        summary = profiling_service.get_profiling_summary(result)

        assert isinstance(summary, dict)
        assert "dataset_size" in summary
        assert "memory_usage" in summary
        assert "feature_breakdown" in summary
        assert "data_quality_score" in summary
        assert "key_insights" in summary
        assert "recommendations" in summary


class TestFeatureEngineeringService:
    """Test suite for FeatureEngineeringService."""

    @pytest.fixture
    def feature_service(self):
        """Create a feature engineering service instance."""
        return FeatureEngineeringService(
            variance_threshold=0.01,
            max_interaction_features=10,
            enable_statistical_features=True,
            enable_interaction_features=True,
        )

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [10.5, 20.1, 30.7, 40.2, 50.8],
                "feature3": [100, 200, 300, 400, 500],
                "categorical1": ["A", "B", "C", "D", "E"],
            }
        )

    @pytest.fixture
    def sample_target(self):
        """Create sample target data."""
        return pd.Series([0, 1, 0, 1, 0], name="target")

    @pytest.mark.asyncio
    async def test_engineer_features_basic(
        self, feature_service, sample_data, sample_target
    ):
        """Test basic feature engineering."""
        result = await feature_service.engineer_features(sample_data, sample_target)

        assert isinstance(result, FeatureEngineeringResult)
        assert isinstance(result.engineered_data, pd.DataFrame)
        assert isinstance(result.selected_features, list)
        assert isinstance(result.engineered_features, list)
        assert isinstance(result.feature_metadata, dict)

        # Check that new features were created
        assert len(result.engineered_features) > 0
        assert result.engineered_data.shape[1] > sample_data.shape[1]

    @pytest.mark.asyncio
    async def test_engineer_features_without_target(self, feature_service, sample_data):
        """Test feature engineering without target variable."""
        result = await feature_service.engineer_features(sample_data)

        assert isinstance(result, FeatureEngineeringResult)
        assert len(result.engineered_features) > 0

    @pytest.mark.asyncio
    async def test_statistical_features_creation(self, feature_service, sample_data):
        """Test statistical features creation."""
        result = await feature_service.engineer_features(sample_data)

        # Check for statistical features
        statistical_features = result.feature_metadata.get("statistical_features", [])
        assert len(statistical_features) > 0

        # Check for specific statistical features
        feature_names = result.engineered_data.columns.tolist()
        assert any("stats_row_mean" in name for name in feature_names)
        assert any("_log" in name for name in feature_names)
        assert any("_squared" in name for name in feature_names)

    @pytest.mark.asyncio
    async def test_interaction_features_creation(self, feature_service, sample_data):
        """Test interaction features creation."""
        result = await feature_service.engineer_features(sample_data)

        # Check for interaction features
        interaction_features = result.feature_metadata.get("interaction_features", [])
        assert len(interaction_features) > 0

        # Check for specific interaction features
        feature_names = result.engineered_data.columns.tolist()
        assert any("_x_" in name for name in feature_names)
        assert any("_plus_" in name for name in feature_names)

    @pytest.mark.asyncio
    async def test_feature_selection(self, feature_service):
        """Test feature selection functionality."""
        # Create data with low variance features
        data_with_low_variance = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [0.001, 0.002, 0.001, 0.002, 0.001],  # Low variance
                "feature3": [100, 200, 300, 400, 500],
            }
        )

        result = await feature_service.engineer_features(data_with_low_variance)

        # Check that feature selection was applied
        assert len(result.selected_features) <= data_with_low_variance.shape[1] + len(
            result.engineered_features
        )
        assert "selection_method" in result.feature_metadata

    @pytest.mark.asyncio
    async def test_feature_importance_scores(
        self, feature_service, sample_data, sample_target
    ):
        """Test feature importance score calculation."""
        result = await feature_service.engineer_features(sample_data, sample_target)

        # Test with target
        importance_scores = feature_service.get_feature_importance_scores(
            result.engineered_data, sample_target
        )

        assert isinstance(importance_scores, dict)
        assert len(importance_scores) > 0

        # Test without target
        importance_scores_no_target = feature_service.get_feature_importance_scores(
            result.engineered_data
        )

        assert isinstance(importance_scores_no_target, dict)
        assert len(importance_scores_no_target) > 0

    @pytest.mark.asyncio
    async def test_feature_summary_creation(self, feature_service, sample_data):
        """Test feature summary creation."""
        result = await feature_service.engineer_features(sample_data)
        summary = feature_service.create_feature_summary(result)

        assert isinstance(summary, dict)
        assert "transformation_summary" in summary
        assert "feature_types" in summary
        assert "selection_info" in summary
        assert "recommendations" in summary

    @pytest.mark.asyncio
    async def test_disabled_features(self, sample_data):
        """Test with disabled feature engineering options."""
        service = FeatureEngineeringService(
            enable_statistical_features=False,
            enable_interaction_features=False,
        )

        result = await service.engineer_features(sample_data)

        # Should have fewer engineered features
        assert len(result.engineered_features) == 0

    @pytest.mark.asyncio
    async def test_error_handling(self, feature_service):
        """Test error handling in feature engineering."""
        # Test with invalid data
        invalid_data = pd.DataFrame({"feature1": [np.inf, -np.inf, np.nan]})

        result = await feature_service.engineer_features(invalid_data)

        # Should not crash and return original data
        assert isinstance(result, FeatureEngineeringResult)
        assert (
            "error" in result.feature_metadata or len(result.engineered_features) >= 0
        )


class TestServiceIntegration:
    """Integration tests for all services working together."""

    @pytest.fixture
    def all_services(self):
        """Create all service instances."""
        return {
            "validation": DataValidationService(min_quality_threshold=0.7),
            "profiling": DataProfilingService(include_advanced_analysis=True),
            "feature_engineering": FeatureEngineeringService(
                variance_threshold=0.01,
                max_interaction_features=5,
            ),
        }

    @pytest.fixture
    def integration_data(self):
        """Create realistic data for integration testing."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "numeric1": np.random.normal(0, 1, 100),
                "numeric2": np.random.exponential(2, 100),
                "categorical1": np.random.choice(["A", "B", "C"], 100),
                "categorical2": np.random.choice(["X", "Y"], 100),
                "boolean1": np.random.choice([True, False], 100),
            }
        )

    @pytest.fixture
    def integration_target(self):
        """Create target for integration testing."""
        np.random.seed(42)
        return pd.Series(np.random.choice([0, 1], 100), name="target")

    @pytest.mark.asyncio
    async def test_full_pipeline_integration(
        self, all_services, integration_data, integration_target
    ):
        """Test full pipeline integration."""
        # Step 1: Validate data
        validation_result = await all_services["validation"].validate_data(
            integration_data, integration_target
        )
        assert isinstance(validation_result, DataValidationResult)

        # Step 2: Profile data
        profiling_result = await all_services["profiling"].profile_data(
            integration_data, integration_target
        )
        assert isinstance(profiling_result, DataProfile)

        # Step 3: Engineer features
        feature_result = await all_services["feature_engineering"].engineer_features(
            integration_data, integration_target
        )
        assert isinstance(feature_result, FeatureEngineeringResult)

        # Verify pipeline progression
        assert profiling_result.basic_stats["n_samples"] == 100
        assert profiling_result.basic_stats["n_features"] == 5
        assert feature_result.engineered_data.shape[0] == 100
        assert feature_result.engineered_data.shape[1] > 5  # Should have more features

    @pytest.mark.asyncio
    async def test_services_with_poor_quality_data(self, all_services):
        """Test services with poor quality data."""
        # Create poor quality data
        poor_data = pd.DataFrame(
            {
                "feature1": [1, 2, np.nan, np.nan, np.nan],
                "feature2": [1, 1, 1, 1, 1],  # Constant
                "feature3": [10, 20, 30, 40, 50],
            }
        )
        # Add duplicates
        poor_data = pd.concat([poor_data, poor_data.iloc[[0, 1]]])

        # Test validation
        validation_result = await all_services["validation"].validate_data(poor_data)
        assert not validation_result.is_valid
        assert validation_result.quality_score < 0.7  # Should be below our threshold

        # Test profiling
        profiling_result = await all_services["profiling"].profile_data(poor_data)
        assert (
            profiling_result.missing_values_ratio > 0.1
        )  # Should have some missing values

        # Test feature engineering (should handle poor data gracefully)
        feature_result = await all_services["feature_engineering"].engineer_features(
            poor_data
        )
        assert isinstance(feature_result, FeatureEngineeringResult)


if __name__ == "__main__":
    # Run basic smoke tests
    async def run_smoke_tests():
        """Run basic smoke tests."""
        print("Running smoke tests for extracted services...")

        # Create sample data
        sample_data = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [10.5, 20.1, 30.7, 40.2, 50.8],
                "feature3": ["A", "B", "C", "D", "E"],
            }
        )
        sample_target = pd.Series([0, 1, 0, 1, 0], name="target")

        # Test validation service
        validation_service = DataValidationService()
        validation_result = await validation_service.validate_data(
            sample_data, sample_target
        )
        print(
            f"✓ Validation service: Quality score = {validation_result.quality_score:.3f}"
        )

        # Test profiling service
        profiling_service = DataProfilingService()
        profiling_result = await profiling_service.profile_data(
            sample_data, sample_target
        )
        print(
            f"✓ Profiling service: Complexity score = {profiling_result.complexity_score:.3f}"
        )

        # Test feature engineering service
        feature_service = FeatureEngineeringService()
        feature_result = await feature_service.engineer_features(
            sample_data, sample_target
        )
        print(
            f"✓ Feature engineering service: {len(sample_data.columns)} -> {len(feature_result.engineered_data.columns)} features"
        )

        print("All smoke tests passed!")

    # Run the smoke tests
    asyncio.run(run_smoke_tests())
