"""Tests for autonomous preprocessing integration."""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from pynomaly.application.services.autonomous_preprocessing import (
    AutonomousPreprocessingOrchestrator,
    AutonomousQualityAnalyzer,
    DataQualityIssue,
    DataQualityReport,
)
from pynomaly.application.services.autonomous_service import (
    AutonomousConfig,
    AutonomousDetectionService,
    DataProfile,
)
from pynomaly.domain.entities import Dataset


@pytest.fixture
def sample_dataset_with_issues():
    """Create a dataset with various data quality issues."""
    np.random.seed(42)

    # Create problematic data
    data = {
        "feature_normal": np.random.normal(0, 1, 100),
        "feature_outliers": np.concatenate(
            [
                np.random.normal(0, 1, 90),
                np.array(
                    [100, -100, 200, -200, 300, -300, 400, -400, 500, -500]
                ),  # 10 extreme outliers
            ]
        ),
        "feature_missing": np.random.normal(0, 1, 100),
        "categorical": np.random.choice(["A", "B", "C"], 100),
        "constant_feature": np.ones(100),
        "high_cardinality": [f"value_{i}" for i in range(100)],  # Every value unique
        "poor_scale_large": np.random.normal(10000, 1000, 100),
        "poor_scale_small": np.random.normal(0.001, 0.0001, 100),
    }

    df = pd.DataFrame(data)

    # Add missing values
    df.loc[10:20, "feature_missing"] = np.nan

    # Add infinite values
    df.loc[0, "feature_outliers"] = np.inf
    df.loc[1, "feature_outliers"] = -np.inf

    # Add duplicates
    df = pd.concat([df, df.iloc[:5]], ignore_index=True)

    return Dataset(
        id="test-dataset-issues",
        name="Test Dataset with Issues",
        description="Dataset with various quality issues for testing",
        data=df,
        metadata={"source": "test"},
    )


@pytest.fixture
def clean_dataset():
    """Create a clean dataset with no quality issues."""
    np.random.seed(42)

    data = {
        "feature1": np.random.normal(0, 1, 100),
        "feature2": np.random.normal(5, 2, 100),
        "feature3": np.random.exponential(1, 100),
        "category": np.random.choice(["A", "B", "C"], 100, p=[0.4, 0.4, 0.2]),
    }

    df = pd.DataFrame(data)

    return Dataset(
        id="test-dataset-clean",
        name="Clean Test Dataset",
        description="Clean dataset with no quality issues",
        data=df,
        metadata={"source": "test"},
    )


class TestAutonomousQualityAnalyzer:
    """Test the quality analyzer component."""

    def test_analyze_clean_data(self, clean_dataset):
        """Test quality analysis on clean data."""
        analyzer = AutonomousQualityAnalyzer()
        report = analyzer.analyze_data_quality(clean_dataset)

        assert isinstance(report, DataQualityReport)
        assert report.overall_score > 0.8  # Should be high quality
        assert not report.preprocessing_required or len(report.issues) == 0
        assert report.estimated_improvement == 0.0 or report.estimated_improvement < 0.2

    def test_analyze_problematic_data(self, sample_dataset_with_issues):
        """Test quality analysis on problematic data."""
        analyzer = AutonomousQualityAnalyzer()
        report = analyzer.analyze_data_quality(sample_dataset_with_issues)

        assert isinstance(report, DataQualityReport)
        assert report.overall_score < 0.8  # Should be low quality
        assert report.preprocessing_required
        assert len(report.issues) > 0
        assert report.estimated_improvement > 0.2

        # Check for expected issue types
        issue_types = {issue.issue_type for issue in report.issues}
        expected_issues = {
            DataQualityIssue.MISSING_VALUES,
            DataQualityIssue.OUTLIERS,
            DataQualityIssue.DUPLICATES,
            DataQualityIssue.CONSTANT_FEATURES,
            DataQualityIssue.INFINITE_VALUES,
            DataQualityIssue.HIGH_CARDINALITY,
            DataQualityIssue.POOR_SCALING,
        }

        assert issue_types.intersection(expected_issues)  # Should detect some of these

    def test_missing_values_detection(self):
        """Test specific missing values detection."""
        # Create data with missing values
        data = pd.DataFrame(
            {
                "col1": [1, 2, np.nan, 4, 5],
                "col2": [1, np.nan, np.nan, 4, 5],
                "col3": [1, 2, 3, 4, 5],  # No missing
            }
        )

        Dataset(id="test", name="test", data=data)
        analyzer = AutonomousQualityAnalyzer()

        missing_issue = analyzer._check_missing_values(data)

        assert missing_issue is not None
        assert missing_issue.issue_type == DataQualityIssue.MISSING_VALUES
        assert "col1" in missing_issue.affected_columns
        assert "col2" in missing_issue.affected_columns
        assert "col3" not in missing_issue.affected_columns
        assert missing_issue.severity > 0

    def test_outlier_detection(self):
        """Test outlier detection."""
        # Create data with clear outliers
        normal_data = np.random.normal(0, 1, 95)
        outliers = np.array([10, -10, 15, -15, 20])
        data = pd.DataFrame({"col1": np.concatenate([normal_data, outliers])})

        analyzer = AutonomousQualityAnalyzer()
        outlier_issue = analyzer._check_outliers(data)

        assert outlier_issue is not None
        assert outlier_issue.issue_type == DataQualityIssue.OUTLIERS
        assert "col1" in outlier_issue.affected_columns
        assert outlier_issue.severity > 0

    def test_duplicates_detection(self):
        """Test duplicate detection."""
        data = pd.DataFrame(
            {
                "col1": [1, 2, 3, 1, 2],  # 2 duplicates
                "col2": [4, 5, 6, 4, 5],
            }
        )

        analyzer = AutonomousQualityAnalyzer()
        duplicate_issue = analyzer._check_duplicates(data)

        assert duplicate_issue is not None
        assert duplicate_issue.issue_type == DataQualityIssue.DUPLICATES
        assert duplicate_issue.metadata["duplicate_count"] == 2

    def test_constant_features_detection(self):
        """Test constant feature detection."""
        data = pd.DataFrame(
            {"constant_col": [1, 1, 1, 1, 1], "variable_col": [1, 2, 3, 4, 5]}
        )

        analyzer = AutonomousQualityAnalyzer()
        constant_issue = analyzer._check_constant_features(data)

        assert constant_issue is not None
        assert constant_issue.issue_type == DataQualityIssue.CONSTANT_FEATURES
        assert "constant_col" in constant_issue.affected_columns
        assert "variable_col" not in constant_issue.affected_columns

    def test_infinite_values_detection(self):
        """Test infinite values detection."""
        data = pd.DataFrame(
            {"col1": [1, 2, np.inf, 4, 5], "col2": [1, 2, 3, -np.inf, 5]}
        )

        analyzer = AutonomousQualityAnalyzer()
        infinite_issue = analyzer._check_infinite_values(data)

        assert infinite_issue is not None
        assert infinite_issue.issue_type == DataQualityIssue.INFINITE_VALUES
        assert "col1" in infinite_issue.affected_columns
        assert "col2" in infinite_issue.affected_columns

    def test_scaling_issues_detection(self):
        """Test scaling issues detection."""
        data = pd.DataFrame(
            {
                "small_scale": np.random.normal(0.001, 0.0001, 100),
                "large_scale": np.random.normal(10000, 1000, 100),
            }
        )

        analyzer = AutonomousQualityAnalyzer()
        scaling_issue = analyzer._check_scaling_issues(data)

        assert scaling_issue is not None
        assert scaling_issue.issue_type == DataQualityIssue.POOR_SCALING
        assert scaling_issue.metadata["scale_ratio"] > 100

    def test_pipeline_recommendation_generation(self, sample_dataset_with_issues):
        """Test pipeline recommendation generation."""
        analyzer = AutonomousQualityAnalyzer()
        report = analyzer.analyze_data_quality(sample_dataset_with_issues)

        assert report.recommended_pipeline is not None
        pipeline = report.recommended_pipeline

        assert "steps" in pipeline
        assert len(pipeline["steps"]) > 0

        # Check that steps are in logical order
        step_operations = [step["operation"] for step in pipeline["steps"]]

        # Infinite values should be handled first if present
        if "handle_infinite_values" in step_operations:
            assert step_operations.index("handle_infinite_values") == 0


class TestAutonomousPreprocessingOrchestrator:
    """Test the preprocessing orchestrator."""

    def test_should_preprocess_clean_data(self, clean_dataset):
        """Test preprocessing decision on clean data."""
        orchestrator = AutonomousPreprocessingOrchestrator()

        should_preprocess, quality_report = orchestrator.should_preprocess(
            clean_dataset, 0.8
        )

        assert not should_preprocess
        assert quality_report.overall_score >= 0.8

    def test_should_preprocess_problematic_data(self, sample_dataset_with_issues):
        """Test preprocessing decision on problematic data."""
        orchestrator = AutonomousPreprocessingOrchestrator()

        should_preprocess, quality_report = orchestrator.should_preprocess(
            sample_dataset_with_issues, 0.8
        )

        assert should_preprocess
        assert (
            quality_report.overall_score < 0.8 or quality_report.preprocessing_required
        )

    def test_preprocess_for_autonomous_detection(self, sample_dataset_with_issues):
        """Test preprocessing application."""
        orchestrator = AutonomousPreprocessingOrchestrator()

        # First assess quality
        should_preprocess, quality_report = orchestrator.should_preprocess(
            sample_dataset_with_issues, 0.8
        )
        assert should_preprocess

        # Apply preprocessing
        processed_dataset, metadata = orchestrator.preprocess_for_autonomous_detection(
            sample_dataset_with_issues, quality_report, max_processing_time=60.0
        )

        assert metadata["preprocessing_applied"]
        assert "applied_steps" in metadata
        assert len(metadata["applied_steps"]) > 0
        assert (
            processed_dataset.data.shape[0] <= sample_dataset_with_issues.data.shape[0]
        )  # May remove rows

        # Check that some issues were fixed
        new_quality_report = orchestrator.quality_analyzer.analyze_data_quality(
            processed_dataset
        )
        assert new_quality_report.overall_score > quality_report.overall_score

    def test_preprocessing_time_limit(self, sample_dataset_with_issues):
        """Test preprocessing time limit enforcement."""
        orchestrator = AutonomousPreprocessingOrchestrator()

        # Get quality report
        should_preprocess, quality_report = orchestrator.should_preprocess(
            sample_dataset_with_issues, 0.8
        )

        # Set a very short time limit
        processed_dataset, metadata = orchestrator.preprocess_for_autonomous_detection(
            sample_dataset_with_issues, quality_report, max_processing_time=0.001
        )

        # Should skip preprocessing due to time constraint
        assert not metadata["preprocessing_applied"]
        assert "Processing time too long" in metadata.get("reason", "")

    def test_preprocessing_error_handling(self, sample_dataset_with_issues):
        """Test error handling in preprocessing."""
        orchestrator = AutonomousPreprocessingOrchestrator()

        # Create a corrupted dataset that will cause errors
        corrupted_data = sample_dataset_with_issues.data.copy()
        corrupted_data.loc[
            0, "feature_normal"
        ] = "not_a_number"  # String in numeric column

        corrupted_dataset = Dataset(
            id="corrupted", name="Corrupted", data=corrupted_data
        )

        should_preprocess, quality_report = orchestrator.should_preprocess(
            corrupted_dataset, 0.8
        )

        # Should handle gracefully even with errors
        processed_dataset, metadata = orchestrator.preprocess_for_autonomous_detection(
            corrupted_dataset, quality_report, max_processing_time=60.0
        )

        # Should return original dataset if preprocessing fails
        assert processed_dataset.id == corrupted_dataset.id


class TestAutonomousServiceIntegration:
    """Test integration with autonomous detection service."""

    @pytest.fixture
    def mock_autonomous_service(self):
        """Create mock autonomous service."""
        mock_detector_repo = Mock()
        mock_result_repo = Mock()
        mock_data_loaders = {}

        service = AutonomousDetectionService(
            detector_repository=mock_detector_repo,
            result_repository=mock_result_repo,
            data_loaders=mock_data_loaders,
        )

        return service

    def test_autonomous_config_preprocessing_options(self):
        """Test autonomous config includes preprocessing options."""
        config = AutonomousConfig(
            enable_preprocessing=True,
            quality_threshold=0.7,
            max_preprocessing_time=120.0,
            preprocessing_strategy="aggressive",
        )

        assert config.enable_preprocessing
        assert config.quality_threshold == 0.7
        assert config.max_preprocessing_time == 120.0
        assert config.preprocessing_strategy == "aggressive"

    @pytest.mark.asyncio
    async def test_assess_and_preprocess_data_disabled(
        self, mock_autonomous_service, clean_dataset
    ):
        """Test preprocessing disabled configuration."""
        config = AutonomousConfig(enable_preprocessing=False)

        (
            processed_dataset,
            profile,
        ) = await mock_autonomous_service._assess_and_preprocess_data(
            clean_dataset, config
        )

        assert processed_dataset.id == clean_dataset.id  # Same dataset
        assert not profile.preprocessing_applied

    @pytest.mark.asyncio
    async def test_assess_and_preprocess_data_enabled(
        self, mock_autonomous_service, sample_dataset_with_issues
    ):
        """Test preprocessing enabled configuration."""
        config = AutonomousConfig(
            enable_preprocessing=True, quality_threshold=0.8, verbose=True
        )

        (
            processed_dataset,
            profile,
        ) = await mock_autonomous_service._assess_and_preprocess_data(
            sample_dataset_with_issues, config
        )

        assert profile.quality_score is not None
        assert profile.quality_report is not None
        assert profile.preprocessing_recommended is not None

        # If preprocessing was applied, check results
        if profile.preprocessing_applied:
            assert profile.preprocessing_metadata is not None
            assert profile.preprocessing_metadata["preprocessing_applied"]

    @pytest.mark.asyncio
    async def test_profile_data_with_preprocessing_info(
        self, mock_autonomous_service, clean_dataset
    ):
        """Test data profiling includes preprocessing information."""
        config = AutonomousConfig()

        # Create initial profile with preprocessing info
        initial_profile = DataProfile(
            n_samples=100,
            n_features=4,
            numeric_features=3,
            categorical_features=1,
            temporal_features=0,
            missing_values_ratio=0.0,
            data_types={},
            correlation_score=0.0,
            sparsity_ratio=0.0,
            outlier_ratio_estimate=0.0,
            seasonality_detected=False,
            trend_detected=False,
            recommended_contamination=0.1,
            complexity_score=0.5,
            quality_score=0.9,
            preprocessing_applied=True,
        )

        final_profile = await mock_autonomous_service._profile_data(
            clean_dataset, config, initial_profile
        )

        # Should preserve preprocessing information
        assert final_profile.quality_score == 0.9
        assert final_profile.preprocessing_applied

    def test_data_profile_preprocessing_fields(self):
        """Test DataProfile includes all preprocessing fields."""
        profile = DataProfile(
            n_samples=100,
            n_features=5,
            numeric_features=3,
            categorical_features=2,
            temporal_features=0,
            missing_values_ratio=0.1,
            data_types={},
            correlation_score=0.5,
            sparsity_ratio=0.0,
            outlier_ratio_estimate=0.05,
            seasonality_detected=False,
            trend_detected=False,
            recommended_contamination=0.1,
            complexity_score=0.6,
            quality_score=0.7,
            preprocessing_recommended=True,
            preprocessing_applied=False,
        )

        assert hasattr(profile, "quality_score")
        assert hasattr(profile, "quality_report")
        assert hasattr(profile, "preprocessing_recommended")
        assert hasattr(profile, "preprocessing_applied")
        assert hasattr(profile, "preprocessing_metadata")

        assert profile.quality_score == 0.7
        assert profile.preprocessing_recommended
        assert not profile.preprocessing_applied


class TestIntegrationWorkflows:
    """Test complete integration workflows."""

    @pytest.mark.asyncio
    async def test_end_to_end_preprocessing_workflow(self, sample_dataset_with_issues):
        """Test complete end-to-end preprocessing workflow."""
        # Setup
        mock_detector_repo = Mock()
        mock_result_repo = Mock()
        mock_data_loaders = {}

        service = AutonomousDetectionService(
            detector_repository=mock_detector_repo,
            result_repository=mock_result_repo,
            data_loaders=mock_data_loaders,
        )

        config = AutonomousConfig(
            enable_preprocessing=True,
            quality_threshold=0.8,
            max_preprocessing_time=60.0,
            verbose=True,
        )

        # Step 1: Load data (simulated)
        original_dataset = sample_dataset_with_issues

        # Step 2: Assess and preprocess
        processed_dataset, initial_profile = await service._assess_and_preprocess_data(
            original_dataset, config
        )

        # Step 3: Profile processed data
        final_profile = await service._profile_data(
            processed_dataset, config, initial_profile
        )

        # Verify workflow
        assert final_profile.quality_score is not None
        assert final_profile.quality_report is not None

        # If preprocessing was applied, verify improvements
        if final_profile.preprocessing_applied:
            assert final_profile.preprocessing_metadata is not None
            assert "applied_steps" in final_profile.preprocessing_metadata

            # Data should be improved
            len(final_profile.quality_report.issues)
            # Note: In a real scenario, we'd compare before/after quality scores

    def test_preprocessing_strategy_selection(self):
        """Test different preprocessing strategies."""
        strategies = ["auto", "aggressive", "conservative", "minimal"]

        for strategy in strategies:
            config = AutonomousConfig(
                preprocessing_strategy=strategy, enable_preprocessing=True
            )

            assert config.preprocessing_strategy == strategy
            assert config.enable_preprocessing

    @pytest.mark.integration
    def test_cli_integration_options(self):
        """Test CLI integration includes preprocessing options."""
        # This would be tested with actual CLI calls in integration tests
        # For now, just verify the config options exist
        config = AutonomousConfig(
            enable_preprocessing=True,
            quality_threshold=0.75,
            max_preprocessing_time=180.0,
            preprocessing_strategy="auto",
        )

        assert hasattr(config, "enable_preprocessing")
        assert hasattr(config, "quality_threshold")
        assert hasattr(config, "max_preprocessing_time")
        assert hasattr(config, "preprocessing_strategy")


if __name__ == "__main__":
    pytest.main([__file__])
