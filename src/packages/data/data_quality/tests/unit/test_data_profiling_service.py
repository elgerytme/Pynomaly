"""Comprehensive unit tests for DataProfilingService."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, call
from uuid import uuid4

from src.data_quality.application.services.data_profiling_service import DataProfilingService
from src.data_quality.domain.entities.data_profile import DataProfile, ColumnProfile, ProfileStatistics, DataType, ProfileStatus
from src.data_quality.infrastructure.adapters.pandas_csv_adapter import PandasCSVAdapter


@pytest.mark.unit
class TestDataProfilingService:
    """Test suite for DataProfilingService."""

    def test_create_profile_basic(self, data_profiling_service, mock_pandas_csv_adapter, sample_csv_data):
        """Test basic profile creation functionality."""
        # Arrange
        dataset_name = "test_dataset"
        source_config = {"file_path": "test.csv"}
        mock_pandas_csv_adapter.read_data.return_value = sample_csv_data
        
        # Act
        profile = data_profiling_service.create_profile(
            dataset_name, mock_pandas_csv_adapter, source_config
        )
        
        # Assert
        assert isinstance(profile, DataProfile)
        assert profile.dataset_name == dataset_name
        assert profile.total_rows == len(sample_csv_data)
        assert profile.total_columns == len(sample_csv_data.columns)
        assert profile.status == ProfileStatus.COMPLETED
        assert len(profile.column_profiles) == len(sample_csv_data.columns)
        
        # Verify repository interaction
        data_profiling_service.data_profile_repository.save.assert_called_once_with(profile)

    def test_create_profile_column_types_inference(self, data_profiling_service, mock_pandas_csv_adapter):
        """Test that column data types are correctly inferred."""
        # Arrange
        test_data = pd.DataFrame({
            'integer_col': [1, 2, 3, 4, 5],
            'float_col': [1.1, 2.2, 3.3, 4.4, 5.5],
            'string_col': ['a', 'b', 'c', 'd', 'e'],
            'boolean_col': [True, False, True, False, True],
            'date_col': pd.date_range('2023-01-01', periods=5)
        })
        mock_pandas_csv_adapter.read_data.return_value = test_data
        
        # Act
        profile = data_profiling_service.create_profile(
            "test_dataset", mock_pandas_csv_adapter, {}
        )
        
        # Assert
        column_types = {cp.column_name: cp.data_type for cp in profile.column_profiles}
        assert column_types['integer_col'] == DataType.INTEGER
        assert column_types['float_col'] == DataType.FLOAT
        assert column_types['string_col'] == DataType.STRING
        assert column_types['boolean_col'] == DataType.BOOLEAN
        assert column_types['date_col'] == DataType.DATE

    def test_create_profile_statistics_calculation(self, data_profiling_service, mock_pandas_csv_adapter):
        """Test that column statistics are correctly calculated."""
        # Arrange
        test_data = pd.DataFrame({
            'col_with_nulls': [1, 2, None, 4, 5, None],
            'col_unique': [1, 2, 3, 4, 5, 6],
            'col_duplicates': [1, 1, 2, 2, 3, 3]
        })
        mock_pandas_csv_adapter.read_data.return_value = test_data
        
        # Act
        profile = data_profiling_service.create_profile(
            "test_dataset", mock_pandas_csv_adapter, {}
        )
        
        # Assert
        stats_by_column = {cp.column_name: cp.statistics for cp in profile.column_profiles}
        
        # Check null counts
        assert stats_by_column['col_with_nulls'].null_count == 2
        assert stats_by_column['col_unique'].null_count == 0
        
        # Check distinct counts
        assert stats_by_column['col_unique'].distinct_count == 6
        assert stats_by_column['col_duplicates'].distinct_count == 3
        
        # Check total counts
        for stats in stats_by_column.values():
            assert stats.total_count == 6

    def test_create_profile_empty_dataset(self, data_profiling_service, mock_pandas_csv_adapter):
        """Test profile creation with empty dataset."""
        # Arrange
        empty_data = pd.DataFrame()
        mock_pandas_csv_adapter.read_data.return_value = empty_data
        
        # Act
        profile = data_profiling_service.create_profile(
            "empty_dataset", mock_pandas_csv_adapter, {}
        )
        
        # Assert
        assert profile.total_rows == 0
        assert profile.total_columns == 0
        assert len(profile.column_profiles) == 0
        assert profile.status == ProfileStatus.COMPLETED

    def test_create_profile_large_dataset(self, data_profiling_service, mock_pandas_csv_adapter, large_dataset):
        """Test profile creation with large dataset."""
        # Arrange
        mock_pandas_csv_adapter.read_data.return_value = large_dataset
        
        # Act
        profile = data_profiling_service.create_profile(
            "large_dataset", mock_pandas_csv_adapter, {}
        )
        
        # Assert
        assert profile.total_rows == len(large_dataset)
        assert profile.total_columns == len(large_dataset.columns)
        assert len(profile.column_profiles) == len(large_dataset.columns)

    def test_create_profile_with_missing_values(self, data_profiling_service, mock_pandas_csv_adapter):
        """Test profile creation with various missing value patterns."""
        # Arrange
        test_data = pd.DataFrame({
            'col1': [1, 2, None, 4, np.nan],
            'col2': ['a', 'b', '', 'd', None],
            'col3': [1.1, 2.2, np.nan, 4.4, None]
        })
        mock_pandas_csv_adapter.read_data.return_value = test_data
        
        # Act
        profile = data_profiling_service.create_profile(
            "test_dataset", mock_pandas_csv_adapter, {}
        )
        
        # Assert
        stats_by_column = {cp.column_name: cp.statistics for cp in profile.column_profiles}
        
        # Check null counts (None and np.nan should be counted)
        assert stats_by_column['col1'].null_count == 2
        assert stats_by_column['col2'].null_count == 1  # Empty string might not be counted as null
        assert stats_by_column['col3'].null_count == 2

    def test_create_profile_adapter_error(self, data_profiling_service, mock_pandas_csv_adapter):
        """Test profile creation when adapter raises an error."""
        # Arrange
        mock_pandas_csv_adapter.read_data.side_effect = Exception("File not found")
        
        # Act & Assert
        with pytest.raises(Exception, match="File not found"):
            data_profiling_service.create_profile(
                "test_dataset", mock_pandas_csv_adapter, {}
            )

    def test_create_profile_repository_error(self, mock_data_profile_repository, mock_pandas_csv_adapter, sample_csv_data):
        """Test profile creation when repository save fails."""
        # Arrange
        service = DataProfilingService(mock_data_profile_repository)
        mock_pandas_csv_adapter.read_data.return_value = sample_csv_data
        mock_data_profile_repository.save.side_effect = Exception("Database error")
        
        # Act & Assert
        with pytest.raises(Exception, match="Database error"):
            service.create_profile("test_dataset", mock_pandas_csv_adapter, {})

    def test_create_profile_with_complex_data_types(self, data_profiling_service, mock_pandas_csv_adapter):
        """Test profile creation with complex data types."""
        # Arrange
        test_data = pd.DataFrame({
            'mixed_types': [1, 'string', 3.14, True, None],
            'json_like': ['{"key": "value"}', '{"key": 123}', None, '{"key": true}', 'invalid_json'],
            'categorical': pd.Categorical(['A', 'B', 'C', 'A', 'B'])
        })
        mock_pandas_csv_adapter.read_data.return_value = test_data
        
        # Act
        profile = data_profiling_service.create_profile(
            "complex_dataset", mock_pandas_csv_adapter, {}
        )
        
        # Assert
        assert len(profile.column_profiles) == 3
        
        # Mixed types should be inferred as string or unknown
        mixed_col = next(cp for cp in profile.column_profiles if cp.column_name == 'mixed_types')
        assert mixed_col.data_type in [DataType.STRING, DataType.UNKNOWN]

    @pytest.mark.parametrize("dataset_size", [10, 100, 1000, 10000])
    def test_create_profile_performance(self, data_profiling_service, mock_pandas_csv_adapter, dataset_size, performance_monitor):
        """Test profile creation performance with different dataset sizes."""
        # Arrange
        test_data = pd.DataFrame({
            'id': range(dataset_size),
            'value': np.random.uniform(0, 100, dataset_size),
            'category': np.random.choice(['A', 'B', 'C'], dataset_size)
        })
        mock_pandas_csv_adapter.read_data.return_value = test_data
        
        # Act
        performance_monitor.start()
        profile = data_profiling_service.create_profile(
            f"dataset_{dataset_size}", mock_pandas_csv_adapter, {}
        )
        metrics = performance_monitor.stop()
        
        # Assert
        assert profile.total_rows == dataset_size
        assert metrics['duration_seconds'] < 10  # Should complete within 10 seconds
        assert metrics['memory_delta_mb'] < 100  # Should not use excessive memory

    def test_create_profile_concurrent_access(self, mock_data_profile_repository, mock_pandas_csv_adapter, sample_csv_data):
        """Test profile creation with concurrent repository access."""
        # Arrange
        service = DataProfilingService(mock_data_profile_repository)
        mock_pandas_csv_adapter.read_data.return_value = sample_csv_data
        
        # Simulate concurrent save calls
        call_count = 0
        def counting_save(profile):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Simulate another save happening concurrently
                service.create_profile("concurrent_dataset", mock_pandas_csv_adapter, {})
        
        mock_data_profile_repository.save.side_effect = counting_save
        
        # Act
        profile = service.create_profile("original_dataset", mock_pandas_csv_adapter, {})
        
        # Assert
        assert profile is not None
        assert call_count == 2  # Original + concurrent

    def test_create_profile_metadata_preservation(self, data_profiling_service, mock_pandas_csv_adapter, sample_csv_data):
        """Test that profile metadata is properly preserved."""
        # Arrange
        dataset_name = "metadata_test_dataset"
        source_config = {"file_path": "test.csv", "delimiter": ",", "encoding": "utf-8"}
        mock_pandas_csv_adapter.read_data.return_value = sample_csv_data
        
        # Act
        profile = data_profiling_service.create_profile(
            dataset_name, mock_pandas_csv_adapter, source_config
        )
        
        # Assert
        assert profile.dataset_name == dataset_name
        assert profile.created_at is not None
        assert profile.updated_at is not None
        assert profile.id is not None

    def test_create_profile_error_recovery(self, data_profiling_service, mock_pandas_csv_adapter):
        """Test profile creation error recovery mechanisms."""
        # Arrange
        # First call fails, second call succeeds
        test_data = pd.DataFrame({'col1': [1, 2, 3]})
        mock_pandas_csv_adapter.read_data.side_effect = [
            Exception("Temporary failure"),
            test_data
        ]
        
        # Act & Assert
        # First call should fail
        with pytest.raises(Exception, match="Temporary failure"):
            data_profiling_service.create_profile(
                "test_dataset", mock_pandas_csv_adapter, {}
            )
        
        # Second call should succeed
        profile = data_profiling_service.create_profile(
            "test_dataset", mock_pandas_csv_adapter, {}
        )
        assert profile is not None
        assert profile.total_rows == 3

    def test_create_profile_source_config_validation(self, data_profiling_service, mock_pandas_csv_adapter, sample_csv_data):
        """Test that source configuration is properly passed to adapter."""
        # Arrange
        source_config = {
            "file_path": "test.csv",
            "delimiter": ";",
            "encoding": "utf-8",
            "skiprows": 1
        }
        mock_pandas_csv_adapter.read_data.return_value = sample_csv_data
        
        # Act
        profile = data_profiling_service.create_profile(
            "test_dataset", mock_pandas_csv_adapter, source_config
        )
        
        # Assert
        mock_pandas_csv_adapter.read_data.assert_called_once_with(source_config)
        assert profile is not None

    def test_create_profile_profile_status_lifecycle(self, data_profiling_service, mock_pandas_csv_adapter, sample_csv_data):
        """Test that profile status follows correct lifecycle."""
        # Arrange
        mock_pandas_csv_adapter.read_data.return_value = sample_csv_data
        
        # Act
        profile = data_profiling_service.create_profile(
            "test_dataset", mock_pandas_csv_adapter, {}
        )
        
        # Assert
        # Profile should start as IN_PROGRESS and end as COMPLETED
        assert profile.status == ProfileStatus.COMPLETED
        assert profile.started_at is not None
        assert profile.completed_at is not None
        assert profile.completed_at >= profile.started_at

    @pytest.mark.slow
    def test_create_profile_memory_efficiency(self, data_profiling_service, mock_pandas_csv_adapter):
        """Test memory efficiency with very large datasets."""
        # Arrange
        large_size = 100000
        test_data = pd.DataFrame({
            'id': range(large_size),
            'data': [f'data_{i}' for i in range(large_size)]
        })
        mock_pandas_csv_adapter.read_data.return_value = test_data
        
        # Act
        import psutil
        import os
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss
        
        profile = data_profiling_service.create_profile(
            "large_memory_test", mock_pandas_csv_adapter, {}
        )
        
        memory_after = process.memory_info().rss
        memory_delta_mb = (memory_after - memory_before) / 1024 / 1024
        
        # Assert
        assert profile.total_rows == large_size
        assert memory_delta_mb < 500  # Should not use more than 500MB additional memory

    def test_create_profile_thread_safety(self, mock_data_profile_repository, mock_pandas_csv_adapter, sample_csv_data):
        """Test thread safety of profile creation."""
        import threading
        import time
        
        # Arrange
        service = DataProfilingService(mock_data_profile_repository)
        mock_pandas_csv_adapter.read_data.return_value = sample_csv_data
        
        results = []
        errors = []
        
        def create_profile_thread(thread_id):
            try:
                profile = service.create_profile(
                    f"thread_dataset_{thread_id}", 
                    mock_pandas_csv_adapter, 
                    {}
                )
                results.append(profile)
            except Exception as e:
                errors.append(e)
        
        # Act
        threads = []
        for i in range(5):
            thread = threading.Thread(target=create_profile_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Assert
        assert len(results) == 5
        assert len(errors) == 0
        assert all(profile is not None for profile in results)


@pytest.mark.integration
class TestDataProfilingServiceIntegration:
    """Integration tests for DataProfilingService with real adapters."""

    def test_create_profile_with_real_csv_adapter(self, mock_data_profile_repository, temp_csv_file):
        """Test profile creation with real CSV adapter."""
        # Arrange
        service = DataProfilingService(mock_data_profile_repository)
        adapter = PandasCSVAdapter()
        source_config = {"file_path": temp_csv_file}
        
        # Act
        profile = service.create_profile("real_csv_test", adapter, source_config)
        
        # Assert
        assert profile is not None
        assert profile.total_rows > 0
        assert profile.total_columns > 0
        assert len(profile.column_profiles) == profile.total_columns

    def test_create_profile_with_corrupted_csv(self, mock_data_profile_repository, corrupted_csv_file):
        """Test profile creation with corrupted CSV data."""
        # Arrange
        service = DataProfilingService(mock_data_profile_repository)
        adapter = PandasCSVAdapter()
        source_config = {"file_path": corrupted_csv_file}
        
        # Act
        profile = service.create_profile("corrupted_csv_test", adapter, source_config)
        
        # Assert
        assert profile is not None
        # Should handle corrupted data gracefully
        assert profile.total_rows > 0
        
        # Check that null counts are properly calculated for corrupted data
        null_counts = [cp.statistics.null_count for cp in profile.column_profiles]
        assert any(count > 0 for count in null_counts)  # Should have some null values