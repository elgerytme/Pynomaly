"""
Comprehensive test suite for Pynomaly Autonomous Mode functionality.
Tests autonomous detection capabilities with various data types and configurations.
"""

import pytest
import asyncio
import tempfile
import csv
import random
import json
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, AsyncMock

@pytest.fixture
def test_data_normal():
    """Create normal tabular test data."""
    temp_dir = Path(tempfile.mkdtemp(prefix="pynomaly_test_"))
    csv_file = temp_dir / "normal_data.csv"
    
    np.random.seed(42)
    
    # Generate normal data
    data = np.random.multivariate_normal([0, 0, 0, 0], np.eye(4), 100)
    df = pd.DataFrame(data, columns=['feature1', 'feature2', 'feature3', 'feature4'])
    df.to_csv(csv_file, index=False)
    
    yield temp_dir, csv_file
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)

@pytest.fixture
def test_data_with_anomalies():
    """Create test data with known anomalies."""
    temp_dir = Path(tempfile.mkdtemp(prefix="pynomaly_test_"))
    csv_file = temp_dir / "anomaly_data.csv"
    
    np.random.seed(42)
    
    # Normal data (80 samples)
    normal_data = np.random.multivariate_normal([0, 0, 0, 0], np.eye(4), 80)
    
    # Anomalous data (20 samples)
    anomaly_data = np.random.uniform(-3, 3, (20, 4))
    
    # Combine data
    data = np.vstack([normal_data, anomaly_data])
    df = pd.DataFrame(data, columns=['feature1', 'feature2', 'feature3', 'feature4'])
    df.to_csv(csv_file, index=False)
    
    yield temp_dir, csv_file
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)

@pytest.fixture
def test_data_high_dimensional():
    """Create high-dimensional test data."""
    temp_dir = Path(tempfile.mkdtemp(prefix="pynomaly_test_"))
    csv_file = temp_dir / "high_dim_data.csv"
    
    np.random.seed(42)
    
    # High-dimensional normal data
    normal_data = np.random.normal(0, 1, (80, 20))
    # High-dimensional anomalies
    anomaly_data = np.random.normal(3, 0.5, (20, 20))
    
    data = np.vstack([normal_data, anomaly_data])
    columns = [f'feature_{i}' for i in range(20)]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(csv_file, index=False)
    
    yield temp_dir, csv_file
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)


class TestAutonomousBasicFunctionality:
    """Test basic autonomous mode functionality."""
    
    def test_imports(self):
        """Test that all required components can be imported."""
        # Test basic scientific libraries
        import pandas as pd
        import numpy as np
        
        # Test pynomaly domain components
        from pynomaly.domain.entities.dataset import Dataset
        from pynomaly.domain.value_objects.contamination_rate import ContaminationRate
        
        # Test autonomous service
        from pynomaly.application.services.autonomous_service import (
            AutonomousDetectionService, 
            AutonomousConfig
        )
        
        # Test data loaders
        from pynomaly.infrastructure.data_loaders.csv_loader import CSVLoader
        
        # Test repositories
        from pynomaly.infrastructure.repositories.in_memory_repositories import (
            InMemoryDetectorRepository,
            InMemoryResultRepository
        )
        
        # All imports successful
        assert True
    
    def test_autonomous_config_creation(self):
        """Test creating autonomous configuration."""
        from pynomaly.application.services.autonomous_service import AutonomousConfig
        
        # Test default config
        config = AutonomousConfig()
        assert config.max_algorithms == 3
        assert config.auto_tune_hyperparams is True
        assert config.confidence_threshold == 0.7
        
        # Test custom config
        custom_config = AutonomousConfig(
            max_algorithms=5,
            auto_tune_hyperparams=False,
            verbose=True
        )
        assert custom_config.max_algorithms == 5
        assert custom_config.auto_tune_hyperparams is False
        assert custom_config.verbose is True
    
    def test_service_initialization(self):
        """Test autonomous service initialization."""
        from pynomaly.application.services.autonomous_service import AutonomousDetectionService
        from pynomaly.infrastructure.data_loaders.csv_loader import CSVLoader
        from pynomaly.infrastructure.repositories.in_memory_repositories import (
            InMemoryDetectorRepository,
            InMemoryResultRepository
        )
        
        # Create service components
        detector_repo = InMemoryDetectorRepository()
        result_repo = InMemoryResultRepository()
        data_loaders = {"csv": CSVLoader()}
        
        # Initialize service
        service = AutonomousDetectionService(
            detector_repository=detector_repo,
            result_repository=result_repo,
            data_loaders=data_loaders
        )
        
        assert service is not None
        assert service.detector_repository == detector_repo
        assert service.result_repository == result_repo
        assert "csv" in service.data_loaders


class TestAutonomousDataProcessing:
    """Test autonomous mode data processing capabilities."""
    
    def test_csv_data_loading(self, test_data_normal):
        """Test loading CSV data."""
        temp_dir, csv_file = test_data_normal
        
        # Load and validate data
        df = pd.read_csv(csv_file)
        
        assert df.shape[0] == 100  # 100 rows
        assert df.shape[1] == 4    # 4 columns
        assert all(df.dtypes == 'float64')  # All numeric
        assert not df.isnull().any().any()  # No missing values
    
    def test_anomaly_data_characteristics(self, test_data_with_anomalies):
        """Test data with known anomalies has expected characteristics."""
        temp_dir, csv_file = test_data_with_anomalies
        
        df = pd.read_csv(csv_file)
        
        assert df.shape[0] == 100  # Total samples
        assert df.shape[1] == 4    # Features
        
        # Check for expected range (anomalies should create wider range)
        data_range = df.max().max() - df.min().min()
        assert data_range > 5  # Should be wider due to anomalies
    
    def test_high_dimensional_data(self, test_data_high_dimensional):
        """Test high-dimensional data characteristics."""
        temp_dir, csv_file = test_data_high_dimensional
        
        df = pd.read_csv(csv_file)
        
        assert df.shape[0] == 100  # Total samples
        assert df.shape[1] == 20   # High-dimensional features
        assert all(col.startswith('feature_') for col in df.columns)


@pytest.mark.asyncio
class TestAutonomousService:
    """Test autonomous detection service functionality."""
    
    async def test_data_loading(self, test_data_normal):
        """Test autonomous data loading."""
        from pynomaly.application.services.autonomous_service import (
            AutonomousDetectionService, 
            AutonomousConfig
        )
        from pynomaly.infrastructure.data_loaders.csv_loader import CSVLoader
        from pynomaly.infrastructure.repositories.in_memory_repositories import (
            InMemoryDetectorRepository,
            InMemoryResultRepository
        )
        
        temp_dir, csv_file = test_data_normal
        
        # Setup service
        service = AutonomousDetectionService(
            detector_repository=InMemoryDetectorRepository(),
            result_repository=InMemoryResultRepository(),
            data_loaders={"csv": CSVLoader()}
        )
        
        config = AutonomousConfig(verbose=False)
        
        # Test data loading
        dataset = await service._auto_load_data(str(csv_file), config)
        
        assert dataset is not None
        assert dataset.name == "normal_data"
        assert dataset.data.shape == (100, 4)
    
    async def test_data_profiling(self, test_data_with_anomalies):
        """Test data profiling functionality."""
        from pynomaly.application.services.autonomous_service import (
            AutonomousDetectionService, 
            AutonomousConfig
        )
        from pynomaly.infrastructure.data_loaders.csv_loader import CSVLoader
        from pynomaly.infrastructure.repositories.in_memory_repositories import (
            InMemoryDetectorRepository,
            InMemoryResultRepository
        )
        
        temp_dir, csv_file = test_data_with_anomalies
        
        # Setup service
        service = AutonomousDetectionService(
            detector_repository=InMemoryDetectorRepository(),
            result_repository=InMemoryResultRepository(),
            data_loaders={"csv": CSVLoader()}
        )
        
        config = AutonomousConfig(verbose=False)
        
        # Load and profile data
        dataset = await service._auto_load_data(str(csv_file), config)
        profile = await service._profile_data(dataset, config)
        
        # Validate profile
        assert profile.n_samples == 100
        assert profile.n_features == 4
        assert profile.numeric_features == 4
        assert profile.categorical_features == 0
        assert 0 <= profile.complexity_score <= 1
        assert 0 <= profile.recommended_contamination <= 1
        assert profile.missing_values_ratio == 0.0
    
    async def test_algorithm_recommendations(self, test_data_high_dimensional):
        """Test algorithm recommendation engine."""
        from pynomaly.application.services.autonomous_service import (
            AutonomousDetectionService, 
            AutonomousConfig
        )
        from pynomaly.infrastructure.data_loaders.csv_loader import CSVLoader
        from pynomaly.infrastructure.repositories.in_memory_repositories import (
            InMemoryDetectorRepository,
            InMemoryResultRepository
        )
        
        temp_dir, csv_file = test_data_high_dimensional
        
        # Setup service
        service = AutonomousDetectionService(
            detector_repository=InMemoryDetectorRepository(),
            result_repository=InMemoryResultRepository(),
            data_loaders={"csv": CSVLoader()}
        )
        
        config = AutonomousConfig(max_algorithms=3, verbose=False)
        
        # Load, profile, and get recommendations
        dataset = await service._auto_load_data(str(csv_file), config)
        profile = await service._profile_data(dataset, config)
        recommendations = await service._recommend_algorithms(profile, config)
        
        # Validate recommendations
        assert len(recommendations) <= 3
        assert len(recommendations) > 0
        
        for rec in recommendations:
            assert hasattr(rec, 'algorithm')
            assert hasattr(rec, 'confidence')
            assert hasattr(rec, 'reasoning')
            assert hasattr(rec, 'expected_performance')
            assert 0 <= rec.confidence <= 1
            assert 0 <= rec.expected_performance <= 1
            assert rec.algorithm is not None
            assert len(rec.reasoning) > 0
    
    @pytest.mark.parametrize("max_algorithms", [1, 2, 3, 5])
    async def test_configuration_options(self, test_data_normal, max_algorithms):
        """Test different configuration options."""
        from pynomaly.application.services.autonomous_service import (
            AutonomousDetectionService, 
            AutonomousConfig
        )
        from pynomaly.infrastructure.data_loaders.csv_loader import CSVLoader
        from pynomaly.infrastructure.repositories.in_memory_repositories import (
            InMemoryDetectorRepository,
            InMemoryResultRepository
        )
        
        temp_dir, csv_file = test_data_normal
        
        # Setup service
        service = AutonomousDetectionService(
            detector_repository=InMemoryDetectorRepository(),
            result_repository=InMemoryResultRepository(),
            data_loaders={"csv": CSVLoader()}
        )
        
        config = AutonomousConfig(
            max_algorithms=max_algorithms,
            auto_tune_hyperparams=False,
            verbose=False
        )
        
        # Test configuration
        dataset = await service._auto_load_data(str(csv_file), config)
        profile = await service._profile_data(dataset, config)
        recommendations = await service._recommend_algorithms(profile, config)
        
        # Verify max_algorithms is respected
        assert len(recommendations) <= max_algorithms


@pytest.mark.integration
class TestAutonomousIntegration:
    """Integration tests for autonomous mode."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self, test_data_with_anomalies):
        """Test complete autonomous detection workflow."""
        from pynomaly.application.services.autonomous_service import (
            AutonomousDetectionService, 
            AutonomousConfig
        )
        from pynomaly.infrastructure.data_loaders.csv_loader import CSVLoader
        from pynomaly.infrastructure.repositories.in_memory_repositories import (
            InMemoryDetectorRepository,
            InMemoryResultRepository
        )
        
        temp_dir, csv_file = test_data_with_anomalies
        
        # Setup complete service
        service = AutonomousDetectionService(
            detector_repository=InMemoryDetectorRepository(),
            result_repository=InMemoryResultRepository(),
            data_loaders={"csv": CSVLoader()}
        )
        
        config = AutonomousConfig(
            max_algorithms=2,
            auto_tune_hyperparams=False,
            verbose=False
        )
        
        # Execute complete workflow
        dataset = await service._auto_load_data(str(csv_file), config)
        profile = await service._profile_data(dataset, config)
        recommendations = await service._recommend_algorithms(profile, config)
        
        # Validate complete workflow
        assert dataset is not None
        assert profile is not None
        assert recommendations is not None
        assert len(recommendations) > 0
        
        # Verify data characteristics are detected correctly
        assert profile.n_samples == 100
        assert profile.n_features == 4
        assert profile.complexity_score > 0
        
        # Verify recommendations are reasonable
        top_recommendation = recommendations[0]
        assert top_recommendation.confidence > 0.5
        assert top_recommendation.algorithm in [
            'LocalOutlierFactor', 
            'IsolationForest', 
            'EllipticEnvelope'
        ]


class TestAutonomousEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_data_handling(self):
        """Test handling of empty datasets."""
        # Create empty dataset
        temp_dir = Path(tempfile.mkdtemp(prefix="pynomaly_test_"))
        csv_file = temp_dir / "empty.csv"
        
        # Create empty CSV with headers only
        df = pd.DataFrame(columns=['feature1', 'feature2'])
        df.to_csv(csv_file, index=False)
        
        try:
            # This should handle empty data gracefully
            df_loaded = pd.read_csv(csv_file)
            assert df_loaded.empty
        finally:
            import shutil
            shutil.rmtree(temp_dir)
    
    def test_single_feature_data(self):
        """Test handling of single-feature datasets."""
        temp_dir = Path(tempfile.mkdtemp(prefix="pynomaly_test_"))
        csv_file = temp_dir / "single_feature.csv"
        
        # Create single-feature data
        data = np.random.normal(0, 1, 100)
        df = pd.DataFrame(data, columns=['feature1'])
        df.to_csv(csv_file, index=False)
        
        try:
            df_loaded = pd.read_csv(csv_file)
            assert df_loaded.shape[1] == 1
            assert df_loaded.shape[0] == 100
        finally:
            import shutil
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])