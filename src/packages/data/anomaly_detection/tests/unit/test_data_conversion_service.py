"""Comprehensive test suite for DataConversionService."""

import pytest
import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json
import gzip
import bz2

from anomaly_detection.domain.services.data_conversion_service import DataConversionService


class TestDataConversionService:
    """Test suite for DataConversionService."""
    
    @pytest.fixture
    def conversion_service(self):
        """Create DataConversionService instance."""
        return DataConversionService()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample test data."""
        np.random.seed(42)
        return pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.randint(0, 100, 100),
            'feature3': np.random.choice(['A', 'B', 'C'], 100),
            'timestamp': pd.date_range('2023-01-01', periods=100, freq='D')
        })
    
    @pytest.fixture
    def temp_csv_file(self, sample_data):
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

    def test_init_supported_formats(self, conversion_service):
        """Test initialization with supported formats."""
        expected_formats = ['csv', 'json', 'parquet', 'excel', 'hdf5', 'pickle']
        
        for fmt in expected_formats:
            assert fmt in conversion_service.supported_formats
        
        assert len(conversion_service.compression_handlers) >= 3
        assert 'gzip' in conversion_service.compression_handlers
        assert 'bz2' in conversion_service.compression_handlers

    @pytest.mark.asyncio
    async def test_convert_file_csv_to_json(self, conversion_service, temp_csv_file, temp_dir):
        """Test conversion from CSV to JSON."""
        output_file = await conversion_service.convert_file(
            input_file=temp_csv_file,
            output_format='json',
            output_dir=temp_dir
        )
        
        assert output_file.exists()
        assert output_file.suffix == '.json'
        
        # Verify content
        with open(output_file) as f:
            data = json.load(f)
            assert len(data) == 100  # Should have 100 records

    @pytest.mark.asyncio
    async def test_convert_file_csv_to_parquet(self, conversion_service, temp_csv_file, temp_dir):
        """Test conversion from CSV to Parquet."""
        output_file = await conversion_service.convert_file(
            input_file=temp_csv_file,
            output_format='parquet',
            output_dir=temp_dir
        )
        
        assert output_file.exists()
        assert output_file.suffix == '.parquet'
        
        # Verify content
        df = pd.read_parquet(output_file)
        assert len(df) == 100

    @pytest.mark.asyncio
    async def test_convert_file_with_compression(self, conversion_service, temp_csv_file, temp_dir):
        """Test file conversion with compression."""
        output_file = await conversion_service.convert_file(
            input_file=temp_csv_file,
            output_format='json',
            output_dir=temp_dir,
            compression='gzip'
        )
        
        assert output_file.exists()
        assert '.gz' in output_file.name
        
        # Verify compressed content
        with gzip.open(output_file, 'rt') as f:
            data = json.load(f)
            assert len(data) == 100

    @pytest.mark.asyncio
    async def test_convert_file_unsupported_format(self, conversion_service, temp_csv_file, temp_dir):
        """Test conversion to unsupported format."""
        with pytest.raises(ValueError, match="Unsupported output format"):
            await conversion_service.convert_file(
                input_file=temp_csv_file,
                output_format='xml',
                output_dir=temp_dir
            )

    @pytest.mark.asyncio
    async def test_convert_file_nonexistent_input(self, conversion_service, temp_dir):
        """Test conversion with nonexistent input file."""
        nonexistent_file = temp_dir / "nonexistent.csv"
        
        with pytest.raises(FileNotFoundError):
            await conversion_service.convert_file(
                input_file=nonexistent_file,
                output_format='json',
                output_dir=temp_dir
            )

    @pytest.mark.asyncio
    async def test_convert_file_with_conversion_options(self, conversion_service, temp_csv_file, temp_dir):
        """Test conversion with custom options."""
        conversion_options = {
            'orient': 'records',
            'date_format': 'iso'
        }
        
        output_file = await conversion_service.convert_file(
            input_file=temp_csv_file,
            output_format='json',
            output_dir=temp_dir,
            conversion_options=conversion_options
        )
        
        assert output_file.exists()

    @pytest.mark.asyncio
    async def test_convert_file_chunked_processing(self, conversion_service, temp_csv_file, temp_dir):
        """Test conversion with chunked processing."""
        output_file = await conversion_service.convert_file(
            input_file=temp_csv_file,
            output_format='json',
            output_dir=temp_dir,
            chunk_size=50  # Process in smaller chunks
        )
        
        assert output_file.exists()
        
        # Verify all data is preserved
        with open(output_file) as f:
            data = json.load(f)
            assert len(data) == 100

    @pytest.mark.asyncio
    async def test_convert_multiple_files(self, conversion_service, sample_data, temp_dir):
        """Test conversion of multiple files."""
        # Create multiple input files
        input_files = []
        for i in range(3):
            file_path = temp_dir / f"input_{i}.csv"
            sample_data.to_csv(file_path, index=False)
            input_files.append(file_path)
        
        output_files = await conversion_service.convert_multiple_files(
            input_files=input_files,
            output_format='json',
            output_dir=temp_dir
        )
        
        assert len(output_files) == 3
        for output_file in output_files:
            assert output_file.exists()
            assert output_file.suffix == '.json'

    @pytest.mark.asyncio
    async def test_convert_directory(self, conversion_service, sample_data, temp_dir):
        """Test conversion of entire directory."""
        # Create input directory with files
        input_dir = temp_dir / "input"
        input_dir.mkdir()
        
        for i in range(3):
            file_path = input_dir / f"data_{i}.csv"
            sample_data.to_csv(file_path, index=False)
        
        output_dir = temp_dir / "output"
        output_files = await conversion_service.convert_directory(
            input_dir=input_dir,
            output_format='parquet',
            output_dir=output_dir,
            pattern="*.csv"
        )
        
        assert len(output_files) >= 3
        assert output_dir.exists()

    def test_to_csv(self, conversion_service, sample_data, temp_dir):
        """Test CSV conversion method."""
        output_file = temp_dir / "test.csv"
        
        conversion_service._to_csv(sample_data, output_file)
        
        assert output_file.exists()
        
        # Verify content
        df = pd.read_csv(output_file)
        assert len(df) == len(sample_data)

    def test_to_json(self, conversion_service, sample_data, temp_dir):
        """Test JSON conversion method."""
        output_file = temp_dir / "test.json"
        
        conversion_service._to_json(sample_data, output_file)
        
        assert output_file.exists()
        
        # Verify content
        with open(output_file) as f:
            data = json.load(f)
            assert len(data) == len(sample_data)

    def test_to_parquet(self, conversion_service, sample_data, temp_dir):
        """Test Parquet conversion method."""
        output_file = temp_dir / "test.parquet"
        
        conversion_service._to_parquet(sample_data, output_file)
        
        assert output_file.exists()
        
        # Verify content
        df = pd.read_parquet(output_file)
        assert len(df) == len(sample_data)

    def test_to_excel(self, conversion_service, sample_data, temp_dir):
        """Test Excel conversion method."""
        output_file = temp_dir / "test.xlsx"
        
        conversion_service._to_excel(sample_data, output_file)
        
        assert output_file.exists()
        
        # Verify content
        df = pd.read_excel(output_file)
        assert len(df) == len(sample_data)

    def test_to_pickle(self, conversion_service, sample_data, temp_dir):
        """Test Pickle conversion method."""
        output_file = temp_dir / "test.pkl"
        
        conversion_service._to_pickle(sample_data, output_file)
        
        assert output_file.exists()
        
        # Verify content
        df = pd.read_pickle(output_file)
        assert len(df) == len(sample_data)

    def test_detect_input_format_csv(self, conversion_service, temp_csv_file):
        """Test input format detection for CSV."""
        format_info = conversion_service._detect_input_format(temp_csv_file)
        
        assert format_info['format'] == 'csv'
        assert 'delimiter' in format_info
        assert 'encoding' in format_info

    def test_detect_input_format_json(self, conversion_service, sample_data, temp_dir):
        """Test input format detection for JSON."""
        json_file = temp_dir / "test.json"
        sample_data.to_json(json_file, orient='records')
        
        format_info = conversion_service._detect_input_format(json_file)
        
        assert format_info['format'] == 'json'

    def test_detect_input_format_unknown(self, conversion_service, temp_dir):
        """Test input format detection for unknown format."""
        unknown_file = temp_dir / "test.unknown"
        unknown_file.write_text("unknown content")
        
        format_info = conversion_service._detect_input_format(unknown_file)
        
        assert format_info['format'] == 'unknown'

    def test_apply_compression_gzip(self, conversion_service, temp_dir):
        """Test GZIP compression."""
        input_file = temp_dir / "test.txt"
        input_file.write_text("test content")
        
        compressed_file = conversion_service._apply_compression(input_file, 'gzip')
        
        assert compressed_file.exists()
        assert '.gz' in compressed_file.name
        
        # Verify compressed content
        with gzip.open(compressed_file, 'rt') as f:
            content = f.read()
            assert content == "test content"

    def test_apply_compression_bz2(self, conversion_service, temp_dir):
        """Test BZ2 compression."""
        input_file = temp_dir / "test.txt"
        input_file.write_text("test content")
        
        compressed_file = conversion_service._apply_compression(input_file, 'bz2')
        
        assert compressed_file.exists()
        assert '.bz2' in compressed_file.name

    def test_apply_compression_unsupported(self, conversion_service, temp_dir):
        """Test unsupported compression format."""
        input_file = temp_dir / "test.txt"
        input_file.write_text("test content")
        
        with pytest.raises(ValueError, match="Unsupported compression"):
            conversion_service._apply_compression(input_file, 'invalid')

    def test_preserve_dtypes_enabled(self, conversion_service, sample_data, temp_dir):
        """Test data type preservation when enabled."""
        output_file = temp_dir / "test.parquet"
        
        # Convert with dtype preservation
        conversion_service._to_parquet(sample_data, output_file, preserve_dtypes=True)
        
        # Verify dtypes are preserved
        df = pd.read_parquet(output_file)
        assert df['feature2'].dtype == sample_data['feature2'].dtype

    def test_preserve_dtypes_disabled(self, conversion_service, sample_data, temp_dir):
        """Test conversion without dtype preservation."""
        output_file = temp_dir / "test.csv"
        
        # Convert without dtype preservation
        conversion_service._to_csv(sample_data, output_file, preserve_dtypes=False)
        
        # CSV will not preserve dtypes exactly
        df = pd.read_csv(output_file)
        assert output_file.exists()

    @pytest.mark.asyncio
    async def test_batch_conversion_with_progress(self, conversion_service, sample_data, temp_dir):
        """Test batch conversion with progress tracking."""
        # Create multiple files
        input_files = []
        for i in range(5):
            file_path = temp_dir / f"batch_{i}.csv"
            sample_data.to_csv(file_path, index=False)
            input_files.append(file_path)
        
        progress_values = []
        def progress_callback(progress):
            progress_values.append(progress)
        
        output_files = await conversion_service.convert_multiple_files(
            input_files=input_files,
            output_format='json',
            output_dir=temp_dir,
            progress_callback=progress_callback
        )
        
        assert len(output_files) == 5
        assert len(progress_values) > 0
        assert progress_values[-1] == 1.0  # Should end at 100%

    def test_memory_efficient_conversion(self, conversion_service, temp_dir):
        """Test memory-efficient conversion for large files."""
        # Create large dataset
        large_data = pd.DataFrame({
            'col1': np.random.rand(10000),
            'col2': np.random.rand(10000)
        })
        
        input_file = temp_dir / "large.csv"
        large_data.to_csv(input_file, index=False)
        
        output_file = temp_dir / "large.parquet"
        
        # Convert with small chunk size for memory efficiency
        conversion_service._to_parquet_chunked(
            input_file, output_file, chunk_size=1000
        )
        
        assert output_file.exists()
        
        # Verify content
        df = pd.read_parquet(output_file)
        assert len(df) == 10000

    def test_schema_preservation(self, conversion_service, sample_data, temp_dir):
        """Test schema preservation across formats."""
        # Convert CSV -> Parquet -> CSV
        csv_file1 = temp_dir / "original.csv"
        parquet_file = temp_dir / "intermediate.parquet"
        csv_file2 = temp_dir / "final.csv"
        
        sample_data.to_csv(csv_file1, index=False)
        
        # CSV -> Parquet
        df1 = pd.read_csv(csv_file1)
        conversion_service._to_parquet(df1, parquet_file)
        
        # Parquet -> CSV
        df2 = pd.read_parquet(parquet_file)
        conversion_service._to_csv(df2, csv_file2)
        
        # Verify schema preservation
        df3 = pd.read_csv(csv_file2)
        assert len(df3) == len(sample_data)
        assert list(df3.columns) == list(sample_data.columns)

    def test_error_handling_corrupted_input(self, conversion_service, temp_dir):
        """Test error handling for corrupted input files."""
        corrupted_file = temp_dir / "corrupted.csv"
        corrupted_file.write_text("malformed,csv\ndata,that,has\ntoo,many,columns,here,and,more")
        
        with pytest.raises(Exception):  # Should raise some parsing error
            df = pd.read_csv(corrupted_file)
            conversion_service._to_json(df, temp_dir / "output.json")

    def test_conversion_metadata(self, conversion_service, sample_data, temp_dir):
        """Test generation of conversion metadata."""
        output_file = temp_dir / "test.json"
        
        metadata = conversion_service._generate_conversion_metadata(
            input_format='csv',
            output_format='json',
            input_size=len(sample_data),
            compression=None
        )
        
        assert 'input_format' in metadata
        assert 'output_format' in metadata
        assert 'conversion_timestamp' in metadata
        assert 'input_records' in metadata
        assert metadata['input_format'] == 'csv'
        assert metadata['output_format'] == 'json'

    @pytest.mark.asyncio
    async def test_concurrent_conversions(self, conversion_service, sample_data, temp_dir):
        """Test concurrent file conversions."""
        # Create multiple input files
        input_files = []
        for i in range(5):
            file_path = temp_dir / f"concurrent_{i}.csv"
            sample_data.to_csv(file_path, index=False)
            input_files.append(file_path)
        
        # Convert all files concurrently
        tasks = []
        for input_file in input_files:
            task = conversion_service.convert_file(
                input_file=input_file,
                output_format='json',
                output_dir=temp_dir
            )
            tasks.append(task)
        
        output_files = await asyncio.gather(*tasks)
        
        assert len(output_files) == 5
        for output_file in output_files:
            assert output_file.exists()

    def test_format_specific_options(self, conversion_service, sample_data, temp_dir):
        """Test format-specific conversion options."""
        # Test CSV with custom delimiter
        csv_file = temp_dir / "custom.csv"
        conversion_service._to_csv(
            sample_data, csv_file, 
            options={'sep': '|', 'index': False}
        )
        
        # Verify custom delimiter
        with open(csv_file) as f:
            first_line = f.readline()
            assert '|' in first_line

    def teardown_method(self):
        """Cleanup after each test."""
        pass