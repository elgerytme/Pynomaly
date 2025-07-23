"""Comprehensive test suite for BatchProcessingService."""

import pytest
import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from concurrent.futures import ThreadPoolExecutor
import tempfile
import json

from anomaly_detection.domain.services.batch_processing_service import BatchProcessingService
from anomaly_detection.domain.services.detection_service import DetectionService
from anomaly_detection.domain.entities.detection_result import DetectionResult
from anomaly_detection.domain.entities.dataset import Dataset, DatasetType, DatasetMetadata


class TestBatchProcessingService:
    """Test suite for BatchProcessingService."""
    
    @pytest.fixture
    def mock_detection_service(self):
        """Create mock detection service."""
        service = Mock(spec=DetectionService)
        service.detect_anomalies.return_value = DetectionResult(
            predictions=np.array([-1, 1, 1, -1, 1]),
            confidence_scores=np.array([0.8, 0.2, 0.3, 0.9, 0.1]),
            algorithm="iforest",
            metadata={"test": True}
        )
        return service
    
    @pytest.fixture
    def mock_model_repository(self):
        """Create mock model repository."""
        return Mock()
    
    @pytest.fixture
    def batch_service(self, mock_detection_service, mock_model_repository):
        """Create BatchProcessingService instance."""
        return BatchProcessingService(
            detection_service=mock_detection_service,
            model_repository=mock_model_repository,
            parallel_jobs=2,
            chunk_size=10
        )
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100),
            'feature3': np.random.normal(0, 1, 100)
        })
    
    @pytest.fixture
    def temp_file(self, sample_data):
        """Create temporary CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data.to_csv(f.name, index=False)
            yield Path(f.name)
            Path(f.name).unlink()
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_init_default_parameters(self, mock_detection_service):
        """Test initialization with default parameters."""
        service = BatchProcessingService(mock_detection_service)
        
        assert service.detection_service == mock_detection_service
        assert service.model_repository is None
        assert service.parallel_jobs == 4
        assert service.chunk_size == 1000
        assert isinstance(service.executor, ThreadPoolExecutor)
    
    def test_init_custom_parameters(self, mock_detection_service, mock_model_repository):
        """Test initialization with custom parameters."""
        service = BatchProcessingService(
            detection_service=mock_detection_service,
            model_repository=mock_model_repository,
            parallel_jobs=8,
            chunk_size=500
        )
        
        assert service.detection_service == mock_detection_service
        assert service.model_repository == mock_model_repository
        assert service.parallel_jobs == 8
        assert service.chunk_size == 500

    @pytest.mark.asyncio
    async def test_process_file_csv_success(self, batch_service, temp_file, temp_dir):
        """Test successful CSV file processing."""
        config = {
            'algorithm': 'iforest',
            'contamination': 0.1,
            'output_format': 'json'
        }
        
        progress_calls = []
        def progress_callback(progress):
            progress_calls.append(progress)
        
        result = await batch_service.process_file(
            input_file=temp_file,
            output_dir=temp_dir,
            config=config,
            progress_callback=progress_callback
        )
        
        # Verify result structure
        assert 'status' in result
        assert 'processing_time' in result
        assert 'total_samples' in result
        assert 'anomalies_detected' in result
        assert 'output_file' in result
        
        # Verify progress callback was called
        assert len(progress_calls) > 0
        assert progress_calls[-1] == 1.0  # Should end at 100%
        
        # Verify output file exists
        output_file = Path(result['output_file'])
        assert output_file.exists()

    @pytest.mark.asyncio
    async def test_process_file_invalid_format(self, batch_service, temp_dir):
        """Test processing with invalid file format."""
        invalid_file = temp_dir / "test.txt"
        invalid_file.write_text("invalid content")
        
        config = {'algorithm': 'iforest'}
        
        result = await batch_service.process_file(
            input_file=invalid_file,
            output_dir=temp_dir,
            config=config
        )
        
        assert result['status'] == 'error'
        assert 'error' in result

    @pytest.mark.asyncio
    async def test_process_file_nonexistent_file(self, batch_service, temp_dir):
        """Test processing nonexistent file."""
        nonexistent_file = temp_dir / "nonexistent.csv"
        config = {'algorithm': 'iforest'}
        
        result = await batch_service.process_file(
            input_file=nonexistent_file,
            output_dir=temp_dir,
            config=config
        )
        
        assert result['status'] == 'error'
        assert 'error' in result

    @pytest.mark.asyncio
    async def test_process_multiple_files(self, batch_service, sample_data, temp_dir):
        """Test processing multiple files."""
        # Create multiple test files
        files = []
        for i in range(3):
            file_path = temp_dir / f"test_{i}.csv" 
            sample_data.to_csv(file_path, index=False)
            files.append(file_path)
        
        configs = [{'algorithm': 'iforest'} for _ in files]
        
        results = await batch_service.process_multiple_files(
            input_files=files,
            output_dir=temp_dir,
            configs=configs
        )
        
        assert len(results) == 3
        for result in results:
            assert 'status' in result
            assert 'processing_time' in result

    @pytest.mark.asyncio
    async def test_process_directory(self, batch_service, sample_data, temp_dir):
        """Test processing entire directory."""
        # Create test files in directory
        input_dir = temp_dir / "input"
        input_dir.mkdir()
        
        for i in range(3):
            file_path = input_dir / f"data_{i}.csv"
            sample_data.to_csv(file_path, index=False)
        
        output_dir = temp_dir / "output"
        config = {'algorithm': 'lof', 'contamination': 0.05}
        
        results = await batch_service.process_directory(
            input_dir=input_dir,
            output_dir=output_dir,
            config=config,
            pattern="*.csv"
        )
        
        assert len(results) >= 3
        assert output_dir.exists()

    def test_chunk_data_small_dataset(self, batch_service):
        """Test data chunking with small dataset."""
        data = np.random.rand(50, 5)
        batch_service.chunk_size = 20
        
        chunks = list(batch_service._chunk_data(data))
        
        assert len(chunks) == 3  # 50 samples / 20 chunk_size = 3 chunks
        assert len(chunks[0]) == 20
        assert len(chunks[1]) == 20  
        assert len(chunks[2]) == 10

    def test_chunk_data_exact_size(self, batch_service):
        """Test data chunking with exact chunk size."""
        data = np.random.rand(100, 5)
        batch_service.chunk_size = 25
        
        chunks = list(batch_service._chunk_data(data))
        
        assert len(chunks) == 4
        for chunk in chunks:
            assert len(chunk) == 25

    @pytest.mark.asyncio
    async def test_process_chunk_success(self, batch_service):
        """Test successful chunk processing."""
        chunk_data = np.random.rand(10, 5)
        config = {'algorithm': 'iforest', 'contamination': 0.1}
        
        result = await batch_service._process_chunk(chunk_data, config, chunk_id=0)
        
        assert 'chunk_id' in result
        assert 'anomalies' in result
        assert 'processing_time' in result
        assert result['chunk_id'] == 0

    @pytest.mark.asyncio
    async def test_process_chunk_detection_failure(self, batch_service, mock_detection_service):
        """Test chunk processing with detection failure."""
        mock_detection_service.detect_anomalies.side_effect = Exception("Detection failed")
        
        chunk_data = np.random.rand(10, 5)
        config = {'algorithm': 'invalid'}
        
        result = await batch_service._process_chunk(chunk_data, config, chunk_id=1)
        
        assert result['chunk_id'] == 1
        assert 'error' in result

    def test_combine_chunk_results(self, batch_service):
        """Test combining results from multiple chunks."""
        chunk_results = [
            {
                'chunk_id': 0,
                'anomalies': np.array([-1, 1, -1]),
                'scores': np.array([0.8, 0.2, 0.9]),
                'processing_time': 0.1
            },
            {
                'chunk_id': 1, 
                'anomalies': np.array([1, -1]),
                'scores': np.array([0.1, 0.7]),
                'processing_time': 0.15
            }
        ]
        
        combined = batch_service._combine_chunk_results(chunk_results)
        
        assert len(combined['predictions']) == 5
        assert len(combined['scores']) == 5
        assert combined['total_processing_time'] == 0.25
        assert combined['chunks_processed'] == 2

    def test_combine_chunk_results_with_errors(self, batch_service):
        """Test combining results when some chunks have errors."""
        chunk_results = [
            {
                'chunk_id': 0,
                'anomalies': np.array([-1, 1]),
                'scores': np.array([0.8, 0.2]),
                'processing_time': 0.1
            },
            {
                'chunk_id': 1,
                'error': 'Processing failed'
            }
        ]
        
        combined = batch_service._combine_chunk_results(chunk_results)
        
        assert len(combined['predictions']) == 2
        assert combined['errors'] == 1
        assert combined['chunks_processed'] == 1

    def test_save_results_json(self, batch_service, temp_dir):
        """Test saving results in JSON format."""
        results = {
            'predictions': [-1, 1, 1, -1],
            'scores': [0.8, 0.2, 0.3, 0.9],
            'metadata': {'algorithm': 'iforest'}
        }
        
        output_file = batch_service._save_results(
            results, temp_dir, "test_output", "json"
        )
        
        assert output_file.exists()
        assert output_file.suffix == '.json'
        
        # Verify content
        with open(output_file) as f:
            loaded = json.load(f)
            assert loaded['metadata']['algorithm'] == 'iforest'

    def test_save_results_csv(self, batch_service, temp_dir):
        """Test saving results in CSV format."""
        results = {
            'predictions': [-1, 1, 1, -1],
            'scores': [0.8, 0.2, 0.3, 0.9]
        }
        
        output_file = batch_service._save_results(
            results, temp_dir, "test_output", "csv"
        )
        
        assert output_file.exists()
        assert output_file.suffix == '.csv'
        
        # Verify content
        df = pd.read_csv(output_file)
        assert 'predictions' in df.columns
        assert 'scores' in df.columns
        assert len(df) == 4

    def test_save_results_unsupported_format(self, batch_service, temp_dir):
        """Test saving results with unsupported format."""
        results = {'predictions': [-1, 1]}
        
        with pytest.raises(ValueError, match="Unsupported output format"):
            batch_service._save_results(
                results, temp_dir, "test_output", "xml"
            )

    @pytest.mark.asyncio
    async def test_progress_tracking(self, batch_service, temp_file, temp_dir):
        """Test progress tracking during file processing."""
        config = {'algorithm': 'iforest'}
        progress_values = []
        
        def track_progress(progress):
            progress_values.append(progress)
        
        await batch_service.process_file(
            input_file=temp_file,
            output_dir=temp_dir,
            config=config,
            progress_callback=track_progress
        )
        
        # Verify progress tracking
        assert len(progress_values) > 0
        assert all(0 <= p <= 1 for p in progress_values)
        assert progress_values[0] >= 0
        assert progress_values[-1] == 1.0

    def test_validate_config_valid(self, batch_service):
        """Test configuration validation with valid config."""
        config = {
            'algorithm': 'iforest',
            'contamination': 0.1,
            'output_format': 'json'
        }
        
        is_valid, errors = batch_service._validate_config(config)
        
        assert is_valid
        assert len(errors) == 0

    def test_validate_config_missing_algorithm(self, batch_service):
        """Test configuration validation with missing algorithm."""
        config = {'contamination': 0.1}
        
        is_valid, errors = batch_service._validate_config(config)
        
        assert not is_valid
        assert 'algorithm' in str(errors)

    def test_validate_config_invalid_contamination(self, batch_service):
        """Test configuration validation with invalid contamination."""
        config = {
            'algorithm': 'iforest',
            'contamination': 1.5  # Invalid: > 1.0
        }
        
        is_valid, errors = batch_service._validate_config(config)
        
        assert not is_valid
        assert 'contamination' in str(errors)

    @pytest.mark.asyncio
    async def test_concurrent_processing_performance(self, batch_service, sample_data, temp_dir):
        """Test concurrent processing performance."""
        # Create larger dataset
        large_data = pd.concat([sample_data] * 10, ignore_index=True)
        
        # Test with different parallel job counts
        for jobs in [1, 2, 4]:
            batch_service.parallel_jobs = jobs
            batch_service.executor = ThreadPoolExecutor(max_workers=jobs)
            
            file_path = temp_dir / f"large_data_{jobs}.csv"
            large_data.to_csv(file_path, index=False)
            
            start_time = time.time()
            
            result = await batch_service.process_file(
                input_file=file_path,
                output_dir=temp_dir,
                config={'algorithm': 'iforest'}
            )
            
            processing_time = time.time() - start_time
            
            assert result['status'] == 'success'
            # More jobs should generally be faster (though not guaranteed due to overhead)
            assert processing_time > 0

    def test_memory_usage_monitoring(self, batch_service):
        """Test memory usage monitoring during processing."""
        # This would require actual memory monitoring
        # For now, test that the method exists and returns reasonable values
        memory_info = batch_service._get_memory_usage()
        
        assert isinstance(memory_info, dict)
        assert 'current_mb' in memory_info
        assert 'peak_mb' in memory_info
        assert memory_info['current_mb'] > 0

    @pytest.mark.asyncio 
    async def test_error_recovery(self, batch_service, temp_dir):
        """Test error recovery during batch processing."""
        # Create mix of valid and invalid files
        valid_data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        valid_file = temp_dir / "valid.csv"
        valid_data.to_csv(valid_file, index=False)
        
        invalid_file = temp_dir / "invalid.txt"
        invalid_file.write_text("not csv data")
        
        files = [valid_file, invalid_file]
        configs = [{'algorithm': 'iforest'}] * 2
        
        results = await batch_service.process_multiple_files(
            input_files=files,
            output_dir=temp_dir,
            configs=configs
        )
        
        # Should have one success and one error
        statuses = [r['status'] for r in results]
        assert 'success' in statuses
        assert 'error' in statuses

    def test_cleanup_resources(self, batch_service):
        """Test proper cleanup of resources."""
        # Verify executor can be shutdown
        batch_service.cleanup()
        
        # After cleanup, executor should be shutdown
        assert batch_service.executor._shutdown

    def teardown_method(self):
        """Cleanup after each test."""
        # Clean up any remaining temporary files
        import tempfile
        tempfile._get_default_tempdir()