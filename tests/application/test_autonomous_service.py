"""Tests for autonomous detection service."""

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from pynomaly.application.services.autonomous_service import (
    AlgorithmRecommendation,
    AutonomousConfig,
    AutonomousDetectionService,
    DataProfile,
)
from pynomaly.domain.entities import Dataset, Detector
from pynomaly.domain.value_objects import AnomalyScore
from pynomaly.infrastructure.data_loaders.csv_loader import CSVLoader
from pynomaly.infrastructure.data_loaders.json_loader import JSONLoader


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    n_samples = 1000

    # Create normal data
    normal_data = np.random.normal(0, 1, (n_samples - 50, 5))

    # Add some anomalies
    anomaly_data = np.random.uniform(-5, 5, (50, 5))

    # Combine
    data = np.vstack([normal_data, anomaly_data])

    # Add categorical column
    categories = np.random.choice(["A", "B", "C"], n_samples)

    df = pd.DataFrame(
        data, columns=["feature1", "feature2", "feature3", "feature4", "feature5"]
    )
    df["category"] = categories
    df["timestamp"] = pd.date_range("2023-01-01", periods=n_samples, freq="1H")

    return df


@pytest.fixture
def mock_detector_repository():
    """Mock detector repository."""
    repo = Mock()
    repo.save = Mock()
    repo.find_by_id = Mock()
    return repo


@pytest.fixture
def mock_result_repository():
    """Mock result repository."""
    repo = Mock()
    repo.save = Mock()
    return repo


@pytest.fixture
def data_loaders():
    """Create data loaders for testing."""
    return {"csv": CSVLoader(), "json": JSONLoader()}


@pytest.fixture
def autonomous_service(mock_detector_repository, mock_result_repository, data_loaders):
    """Create autonomous service for testing."""
    return AutonomousDetectionService(
        detector_repository=mock_detector_repository,
        result_repository=mock_result_repository,
        data_loaders=data_loaders,
    )


class TestAutonomousConfig:
    """Test autonomous configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AutonomousConfig()

        assert config.max_samples_analysis == 10000
        assert config.confidence_threshold == 0.8
        assert config.max_algorithms == 5
        assert config.auto_tune_hyperparams is True
        assert config.save_results is True
        assert config.export_results is False
        assert config.export_format == "csv"
        assert config.verbose is False

    def test_custom_config(self):
        """Test custom configuration."""
        config = AutonomousConfig(
            max_samples_analysis=5000,
            confidence_threshold=0.9,
            max_algorithms=3,
            auto_tune_hyperparams=False,
            verbose=True,
        )

        assert config.max_samples_analysis == 5000
        assert config.confidence_threshold == 0.9
        assert config.max_algorithms == 3
        assert config.auto_tune_hyperparams is False
        assert config.verbose is True


class TestDataProfile:
    """Test data profiling functionality."""

    @pytest.mark.asyncio
    async def test_profile_data(self, autonomous_service, sample_dataframe):
        """Test data profiling."""
        dataset = Dataset(name="test_data", data=sample_dataframe)

        config = AutonomousConfig()
        profile = await autonomous_service._profile_data(dataset, config)

        assert isinstance(profile, DataProfile)
        assert profile.n_samples == 1000
        assert profile.n_features == 7  # 5 numeric + 1 categorical + 1 timestamp
        assert profile.numeric_features == 5
        assert profile.categorical_features == 1
        assert profile.temporal_features == 1
        assert profile.missing_values_ratio == 0.0  # No missing values
        assert 0 <= profile.correlation_score <= 1
        assert 0 <= profile.complexity_score <= 1
        assert 0 <= profile.recommended_contamination <= 1

    @pytest.mark.asyncio
    async def test_profile_data_with_missing_values(self, autonomous_service):
        """Test profiling data with missing values."""
        # Create data with missing values
        df = pd.DataFrame(
            {
                "feature1": [1, 2, np.nan, 4, 5],
                "feature2": [1, np.nan, 3, 4, np.nan],
                "category": ["A", "B", None, "C", "A"],
            }
        )

        dataset = Dataset(name="test_missing", data=df)
        config = AutonomousConfig()

        profile = await autonomous_service._profile_data(dataset, config)

        assert profile.missing_values_ratio > 0
        assert profile.n_samples == 5
        assert profile.n_features == 3

    @pytest.mark.asyncio
    async def test_profile_large_data_sampling(self, autonomous_service):
        """Test that large datasets are sampled for profiling."""
        # Create large dataset
        np.random.seed(42)
        large_df = pd.DataFrame(
            np.random.normal(0, 1, (20000, 10)),
            columns=[f"feature_{i}" for i in range(10)],
        )

        dataset = Dataset(name="large_data", data=large_df)
        config = AutonomousConfig(max_samples_analysis=5000)

        profile = await autonomous_service._profile_data(dataset, config)

        # Original data has 20000 samples, but profiling should still work
        assert profile.n_samples == 20000
        assert profile.n_features == 10


class TestAlgorithmRecommendation:
    """Test algorithm recommendation system."""

    @pytest.mark.asyncio
    async def test_recommend_algorithms_general(self, autonomous_service):
        """Test algorithm recommendations for general dataset."""
        profile = DataProfile(
            n_samples=1000,
            n_features=10,
            numeric_features=8,
            categorical_features=2,
            temporal_features=0,
            missing_values_ratio=0.1,
            data_types={"feature1": "float64"},
            correlation_score=0.3,
            sparsity_ratio=0.05,
            outlier_ratio_estimate=0.05,
            seasonality_detected=False,
            trend_detected=False,
            recommended_contamination=0.05,
            complexity_score=0.4,
        )

        config = AutonomousConfig()
        recommendations = await autonomous_service._recommend_algorithms(
            profile, config
        )

        assert isinstance(recommendations, list)
        assert len(recommendations) <= config.max_algorithms
        assert all(isinstance(rec, AlgorithmRecommendation) for rec in recommendations)

        # Should include IsolationForest as it's general purpose
        algorithms = [rec.algorithm for rec in recommendations]
        assert "IsolationForest" in algorithms

        # Check that recommendations are sorted by confidence
        confidences = [rec.confidence for rec in recommendations]
        assert confidences == sorted(confidences, reverse=True)

    @pytest.mark.asyncio
    async def test_recommend_algorithms_numeric_data(self, autonomous_service):
        """Test recommendations for mostly numeric data."""
        profile = DataProfile(
            n_samples=5000,
            n_features=15,
            numeric_features=14,  # Mostly numeric
            categorical_features=1,
            temporal_features=0,
            missing_values_ratio=0.02,
            data_types={"feature1": "float64"},
            correlation_score=0.2,
            sparsity_ratio=0.01,
            outlier_ratio_estimate=0.03,
            seasonality_detected=False,
            trend_detected=False,
            recommended_contamination=0.03,
            complexity_score=0.3,
        )

        config = AutonomousConfig()
        recommendations = await autonomous_service._recommend_algorithms(
            profile, config
        )

        algorithms = [rec.algorithm for rec in recommendations]

        # Should recommend LOF for numeric data
        assert "LocalOutlierFactor" in algorithms
        assert "IsolationForest" in algorithms

    @pytest.mark.asyncio
    async def test_recommend_algorithms_complex_data(self, autonomous_service):
        """Test recommendations for complex dataset."""
        profile = DataProfile(
            n_samples=50000,
            n_features=100,
            numeric_features=80,
            categorical_features=20,
            temporal_features=0,
            missing_values_ratio=0.15,
            data_types={"feature1": "float64"},
            correlation_score=0.7,  # High correlation
            sparsity_ratio=0.3,
            outlier_ratio_estimate=0.08,
            seasonality_detected=False,
            trend_detected=False,
            recommended_contamination=0.08,
            complexity_score=0.8,  # High complexity
        )

        config = AutonomousConfig()
        recommendations = await autonomous_service._recommend_algorithms(
            profile, config
        )

        algorithms = [rec.algorithm for rec in recommendations]

        # Should recommend deep learning for complex, large data
        assert "AutoEncoder" in algorithms


class TestDataSourceDetection:
    """Test automatic data source detection."""

    def test_detect_csv_format(self, autonomous_service):
        """Test CSV format detection."""
        # Test by extension
        assert autonomous_service._detect_data_format(Path("data.csv")) == "csv"
        assert autonomous_service._detect_data_format(Path("data.tsv")) == "csv"
        assert autonomous_service._detect_data_format(Path("data.txt")) == "csv"

    def test_detect_json_format(self, autonomous_service):
        """Test JSON format detection."""
        assert autonomous_service._detect_data_format(Path("data.json")) == "json"
        assert autonomous_service._detect_data_format(Path("data.jsonl")) == "json"

    def test_detect_excel_format(self, autonomous_service):
        """Test Excel format detection."""
        assert autonomous_service._detect_data_format(Path("data.xlsx")) == "excel"
        assert autonomous_service._detect_data_format(Path("data.xls")) == "excel"

    def test_detect_parquet_format(self, autonomous_service):
        """Test Parquet format detection."""
        assert autonomous_service._detect_data_format(Path("data.parquet")) == "parquet"
        assert autonomous_service._detect_data_format(Path("data.pq")) == "parquet"


class TestDetectorCreation:
    """Test detector creation from recommendations."""

    def test_create_detector(self, autonomous_service, sample_dataframe):
        """Test detector creation."""
        recommendation = AlgorithmRecommendation(
            algorithm="IsolationForest",
            confidence=0.85,
            reasoning="Good general purpose algorithm",
            hyperparams={
                "n_estimators": 100,
                "contamination": 0.05,
                "random_state": 42,
            },
            expected_performance=0.75,
        )

        dataset = Dataset(name="test_data", data=sample_dataframe)
        detector = autonomous_service._create_detector(recommendation, dataset)

        assert isinstance(detector, Detector)
        assert detector.algorithm == "IsolationForest"
        assert detector.name == "auto_isolationforest"
        assert detector.hyperparams["n_estimators"] == 100
        assert detector.hyperparams["contamination"] == 0.05
        assert detector.metadata["auto_generated"] is True
        assert detector.metadata["confidence"] == 0.85


class TestAutonomousDetection:
    """Test full autonomous detection workflow."""

    @pytest.mark.asyncio
    async def test_detect_autonomous_with_dataframe(
        self, autonomous_service, sample_dataframe
    ):
        """Test autonomous detection with DataFrame input."""
        config = AutonomousConfig(
            max_algorithms=2, auto_tune_hyperparams=False, save_results=False
        )

        # Mock the internal methods to avoid complex detector creation
        with patch.object(
            autonomous_service, "_run_detection_pipeline"
        ) as mock_pipeline:
            mock_pipeline.return_value = {
                "IsolationForest": Mock(
                    n_anomalies=50,
                    anomaly_rate=0.05,
                    threshold=0.5,
                    execution_time_ms=100,
                    scores=[AnomalyScore(0.1), AnomalyScore(0.9)],
                    labels=[0, 1],
                    anomalies=[],
                    score_statistics={"mean": 0.5},
                )
            }

            results = await autonomous_service.detect_autonomous(
                sample_dataframe, config
            )

            assert "autonomous_detection_results" in results
            auto_results = results["autonomous_detection_results"]
            assert auto_results["success"] is True
            assert "data_profile" in auto_results
            assert "algorithm_recommendations" in auto_results
            assert "detection_results" in auto_results

    @pytest.mark.asyncio
    async def test_autonomous_detection_error_handling(self, autonomous_service):
        """Test error handling in autonomous detection."""
        config = AutonomousConfig()

        # Test with invalid data source
        with pytest.raises(Exception):
            await autonomous_service.detect_autonomous("nonexistent_file.csv", config)


class TestConfigurationOptimization:
    """Test automatic configuration optimization."""

    def test_auto_configure_csv_loader(self, autonomous_service, tmp_path):
        """Test auto-configuration for CSV loader."""
        # Create test CSV with different delimiter
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("col1;col2;col3\n1;2;3\n4;5;6\n")

        options = autonomous_service._auto_configure_loader(csv_file, "csv")

        # Should detect semicolon delimiter
        assert options.get("delimiter") == ";"

    def test_auto_configure_csv_encoding(self, autonomous_service, tmp_path):
        """Test encoding detection for CSV."""
        # Create test CSV with UTF-8 content
        csv_file = tmp_path / "test_utf8.csv"
        with open(csv_file, "w", encoding="utf-8") as f:
            f.write("name,value\ntest,123\nüñíçødé,456\n")

        options = autonomous_service._auto_configure_loader(csv_file, "csv")

        # Should work with default configuration
        assert isinstance(options, dict)


@pytest.mark.asyncio
async def test_full_integration(tmp_path):
    """Integration test for full autonomous detection workflow."""
    # Create test CSV file
    test_data = pd.DataFrame(
        {
            "feature1": np.random.normal(0, 1, 100),
            "feature2": np.random.normal(0, 1, 100),
            "feature3": np.random.choice(["A", "B", "C"], 100),
        }
    )

    # Add a few anomalies
    test_data.loc[95:99, "feature1"] = 10  # Clear anomalies

    csv_file = tmp_path / "test_data.csv"
    test_data.to_csv(csv_file, index=False)

    # Setup mocks
    mock_detector_repo = Mock()
    mock_result_repo = Mock()

    data_loaders = {"csv": CSVLoader(), "json": JSONLoader()}

    service = AutonomousDetectionService(
        detector_repository=mock_detector_repo,
        result_repository=mock_result_repo,
        data_loaders=data_loaders,
    )

    config = AutonomousConfig(
        max_algorithms=1, auto_tune_hyperparams=False, save_results=False
    )

    # Mock the pipeline to avoid complex detector dependencies
    with patch.object(service, "_run_detection_pipeline") as mock_pipeline:
        # Create a simple mock result
        mock_result = Mock()
        mock_result.n_anomalies = 5
        mock_result.anomaly_rate = 0.05
        mock_result.threshold = 0.5
        mock_result.execution_time_ms = 50
        mock_result.scores = [AnomalyScore(0.1)] * 100
        mock_result.labels = [0] * 95 + [1] * 5
        mock_result.anomalies = []
        mock_result.score_statistics = {"mean": 0.5, "std": 0.2}

        mock_pipeline.return_value = {"IsolationForest": mock_result}

        # Run autonomous detection
        results = await service.detect_autonomous(str(csv_file), config)

        # Verify results structure
        assert "autonomous_detection_results" in results
        auto_results = results["autonomous_detection_results"]

        assert auto_results["success"] is True
        assert "data_profile" in auto_results
        assert "algorithm_recommendations" in auto_results
        assert "detection_results" in auto_results
        assert auto_results["best_algorithm"] == "IsolationForest"

        # Verify data profile
        profile = auto_results["data_profile"]
        assert profile["samples"] == 100
        assert profile["features"] == 3
        assert profile["numeric_features"] == 2

        # Verify detection results
        detection_results = auto_results["detection_results"]["IsolationForest"]
        assert detection_results["anomalies_found"] == 5
        assert detection_results["anomaly_rate"] == 0.05
