import pytest
import pandas as pd
import numpy as np
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, mock_open

from src.packages.data_profiling.infrastructure.adapters.file_adapter import FileAdapter, MultiFileAdapter


class TestFileAdapter:
    """Test FileAdapter class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def sample_csv_data(self):
        """Sample CSV data for testing."""
        return pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'age': [25, 30, 35, 40, 45],
            'salary': [50000, 60000, 70000, 80000, 90000],
            'is_active': [True, False, True, True, False]
        })
    
    @pytest.fixture
    def sample_json_data(self):
        """Sample JSON data for testing."""
        return [
            {'id': 1, 'name': 'Alice', 'age': 25},
            {'id': 2, 'name': 'Bob', 'age': 30},
            {'id': 3, 'name': 'Charlie', 'age': 35}
        ]


class TestFileAdapterInitialization:
    """Test FileAdapter initialization."""
    
    def test_adapter_initialization(self, temp_dir):
        """Test FileAdapter initialization."""
        test_file = Path(temp_dir) / "test.csv"
        test_file.touch()  # Create empty file
        
        adapter = FileAdapter(str(test_file))
        
        assert adapter.file_path == test_file
        assert adapter.kwargs == {}
        assert adapter.detected_encoding is None
        assert adapter.detected_separator is None
    
    def test_adapter_initialization_with_kwargs(self, temp_dir):
        """Test FileAdapter initialization with kwargs."""
        test_file = Path(temp_dir) / "test.csv"
        test_file.touch()
        
        kwargs = {'encoding': 'utf-8', 'sep': ';'}
        adapter = FileAdapter(str(test_file), **kwargs)
        
        assert adapter.kwargs == kwargs
    
    def test_adapter_initialization_nonexistent_file(self):
        """Test FileAdapter initialization with non-existent file."""
        adapter = FileAdapter("/nonexistent/file.csv")
        
        assert adapter.file_path == Path("/nonexistent/file.csv")


class TestCSVLoading:
    """Test CSV file loading functionality."""
    
    def test_load_csv_basic(self, temp_dir, sample_csv_data):
        """Test basic CSV loading."""
        csv_file = Path(temp_dir) / "test.csv"
        sample_csv_data.to_csv(csv_file, index=False)
        
        adapter = FileAdapter(str(csv_file))
        result = adapter.load_data()
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_csv_data)
        assert list(result.columns) == list(sample_csv_data.columns)
        pd.testing.assert_frame_equal(result, sample_csv_data)
    
    def test_load_csv_with_custom_separator(self, temp_dir, sample_csv_data):
        """Test CSV loading with custom separator."""
        csv_file = Path(temp_dir) / "test.csv"
        sample_csv_data.to_csv(csv_file, index=False, sep=';')
        
        adapter = FileAdapter(str(csv_file))
        result = adapter.load_data()
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_csv_data)
    
    def test_load_csv_with_encoding(self, temp_dir):
        """Test CSV loading with specific encoding."""
        csv_file = Path(temp_dir) / "test.csv"
        
        # Create CSV with special characters
        data = pd.DataFrame({
            'name': ['José', 'François', 'München'],
            'value': [1, 2, 3]
        })
        data.to_csv(csv_file, index=False, encoding='utf-8')
        
        adapter = FileAdapter(str(csv_file))
        result = adapter.load_data()
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
    
    def test_load_csv_fallback_on_detection_error(self, temp_dir, sample_csv_data):
        """Test CSV loading fallback when detection fails."""
        csv_file = Path(temp_dir) / "test.csv"
        sample_csv_data.to_csv(csv_file, index=False)
        
        adapter = FileAdapter(str(csv_file))
        
        # Mock detection methods to raise exceptions
        with patch.object(adapter, '_detect_encoding', side_effect=Exception("Detection error")), \
             patch.object(adapter, '_detect_csv_separator', side_effect=Exception("Detection error")):
            
            result = adapter.load_data()
            
            assert isinstance(result, pd.DataFrame)
            assert len(result) == len(sample_csv_data)
    
    def test_load_csv_file_not_found(self):
        """Test CSV loading with non-existent file."""
        adapter = FileAdapter("/nonexistent/file.csv")
        
        with pytest.raises(FileNotFoundError):
            adapter.load_data()


class TestJSONLoading:
    """Test JSON file loading functionality."""
    
    def test_load_json_array(self, temp_dir, sample_json_data):
        """Test loading JSON array."""
        json_file = Path(temp_dir) / "test.json"
        
        with open(json_file, 'w') as f:
            json.dump(sample_json_data, f)
        
        adapter = FileAdapter(str(json_file))
        result = adapter.load_data()
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_json_data)
        assert 'id' in result.columns
        assert 'name' in result.columns
        assert 'age' in result.columns
    
    def test_load_json_single_object(self, temp_dir):
        """Test loading single JSON object."""
        json_file = Path(temp_dir) / "test.json"
        
        single_object = {'id': 1, 'name': 'Alice', 'age': 25}
        with open(json_file, 'w') as f:
            json.dump(single_object, f)
        
        adapter = FileAdapter(str(json_file))
        result = adapter.load_data()
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result.iloc[0]['name'] == 'Alice'
    
    def test_load_jsonl(self, temp_dir, sample_json_data):
        """Test loading JSONL (JSON Lines) format."""
        jsonl_file = Path(temp_dir) / "test.jsonl"
        
        with open(jsonl_file, 'w') as f:
            for item in sample_json_data:
                json.dump(item, f)
                f.write('\n')
        
        adapter = FileAdapter(str(jsonl_file))
        result = adapter.load_data()
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_json_data)
    
    def test_load_json_invalid_structure(self, temp_dir):
        """Test loading invalid JSON structure."""
        json_file = Path(temp_dir) / "test.json"
        
        with open(json_file, 'w') as f:
            json.dump("invalid_structure", f)
        
        adapter = FileAdapter(str(json_file))
        
        with pytest.raises(ValueError, match="Unsupported JSON structure"):
            adapter.load_data()


class TestParquetLoading:
    """Test Parquet file loading functionality."""
    
    def test_load_parquet(self, temp_dir, sample_csv_data):
        """Test Parquet loading."""
        parquet_file = Path(temp_dir) / "test.parquet"
        sample_csv_data.to_parquet(parquet_file, index=False)
        
        adapter = FileAdapter(str(parquet_file))
        result = adapter.load_data()
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_csv_data)
        assert list(result.columns) == list(sample_csv_data.columns)


class TestExcelLoading:
    """Test Excel file loading functionality."""
    
    def test_load_excel(self, temp_dir, sample_csv_data):
        """Test Excel loading."""
        excel_file = Path(temp_dir) / "test.xlsx"
        sample_csv_data.to_excel(excel_file, index=False)
        
        adapter = FileAdapter(str(excel_file))
        result = adapter.load_data()
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_csv_data)
    
    def test_load_excel_specific_sheet(self, temp_dir, sample_csv_data):
        """Test Excel loading with specific sheet."""
        excel_file = Path(temp_dir) / "test.xlsx"
        
        with pd.ExcelWriter(excel_file) as writer:
            sample_csv_data.to_excel(writer, sheet_name='Sheet1', index=False)
            sample_csv_data.to_excel(writer, sheet_name='Sheet2', index=False)
        
        adapter = FileAdapter(str(excel_file), sheet_name='Sheet2')
        result = adapter.load_data()
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_csv_data)


class TestSpecializedFormats:
    """Test loading of specialized formats."""
    
    def test_load_feather(self, temp_dir, sample_csv_data):
        """Test Feather loading."""
        feather_file = Path(temp_dir) / "test.feather"
        sample_csv_data.to_feather(feather_file)
        
        adapter = FileAdapter(str(feather_file))
        result = adapter.load_data()
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_csv_data)
    
    def test_load_pickle(self, temp_dir, sample_csv_data):
        """Test Pickle loading."""
        pickle_file = Path(temp_dir) / "test.pkl"
        sample_csv_data.to_pickle(pickle_file)
        
        adapter = FileAdapter(str(pickle_file))
        result = adapter.load_data()
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_csv_data)
    
    def test_load_pickle_invalid_content(self, temp_dir):
        """Test Pickle loading with invalid content."""
        pickle_file = Path(temp_dir) / "test.pkl"
        
        # Save non-DataFrame object
        with open(pickle_file, 'wb') as f:
            import pickle
            pickle.dump("not_a_dataframe", f)
        
        adapter = FileAdapter(str(pickle_file))
        
        with pytest.raises(ValueError, match="Pickle file does not contain a DataFrame"):
            adapter.load_data()
    
    def test_load_tsv(self, temp_dir, sample_csv_data):
        """Test TSV loading."""
        tsv_file = Path(temp_dir) / "test.tsv"
        sample_csv_data.to_csv(tsv_file, index=False, sep='\t')
        
        adapter = FileAdapter(str(tsv_file))
        result = adapter.load_data()
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_csv_data)
    
    def test_load_text_file(self, temp_dir):
        """Test text file loading."""
        text_file = Path(temp_dir) / "test.txt"
        
        lines = ["Line 1", "Line 2", "Line 3"]
        with open(text_file, 'w') as f:
            f.write('\n'.join(lines))
        
        adapter = FileAdapter(str(text_file))
        result = adapter.load_data()
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(lines)
        assert 'text' in result.columns
        assert result['text'].tolist() == lines


class TestEncodingDetection:
    """Test encoding detection functionality."""
    
    def test_detect_encoding_utf8(self, temp_dir):
        """Test UTF-8 encoding detection."""
        test_file = Path(temp_dir) / "test.csv"
        
        # Create file with UTF-8 content
        content = "id,name\n1,José\n2,François"
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        adapter = FileAdapter(str(test_file))
        encoding = adapter._detect_encoding()
        
        assert encoding in ['utf-8', 'UTF-8']
    
    def test_detect_encoding_fallback(self, temp_dir):
        """Test encoding detection fallback."""
        test_file = Path(temp_dir) / "test.csv"
        test_file.touch()
        
        adapter = FileAdapter(str(test_file))
        
        # Mock chardet to return None
        with patch('chardet.detect', return_value={'encoding': None, 'confidence': 0}):
            encoding = adapter._detect_encoding()
            
            assert encoding == 'utf-8'  # Should fallback to utf-8
    
    def test_detect_encoding_error_handling(self):
        """Test encoding detection error handling."""
        adapter = FileAdapter("/nonexistent/file.csv")
        
        encoding = adapter._detect_encoding()
        
        assert encoding == 'utf-8'  # Should fallback to utf-8


class TestSeparatorDetection:
    """Test CSV separator detection functionality."""
    
    def test_detect_csv_separator_comma(self, temp_dir):
        """Test comma separator detection."""
        csv_file = Path(temp_dir) / "test.csv"
        
        content = "id,name,age\n1,Alice,25\n2,Bob,30"
        with open(csv_file, 'w') as f:
            f.write(content)
        
        adapter = FileAdapter(str(csv_file))
        separator = adapter._detect_csv_separator()
        
        assert separator == ','
    
    def test_detect_csv_separator_semicolon(self, temp_dir):
        """Test semicolon separator detection."""
        csv_file = Path(temp_dir) / "test.csv"
        
        content = "id;name;age\n1;Alice;25\n2;Bob;30"
        with open(csv_file, 'w') as f:
            f.write(content)
        
        adapter = FileAdapter(str(csv_file))
        separator = adapter._detect_csv_separator()
        
        assert separator == ';'
    
    def test_detect_csv_separator_tab(self, temp_dir):
        """Test tab separator detection."""
        csv_file = Path(temp_dir) / "test.csv"
        
        content = "id\tname\tage\n1\tAlice\t25\n2\tBob\t30"
        with open(csv_file, 'w') as f:
            f.write(content)
        
        adapter = FileAdapter(str(csv_file))
        separator = adapter._detect_csv_separator()
        
        assert separator == '\t'
    
    def test_detect_csv_separator_fallback(self, temp_dir):
        """Test separator detection fallback."""
        csv_file = Path(temp_dir) / "test.csv"
        
        # Inconsistent separators
        content = "id,name\n1;Alice\n2:Bob"
        with open(csv_file, 'w') as f:
            f.write(content)
        
        adapter = FileAdapter(str(csv_file))
        separator = adapter._detect_csv_separator()
        
        assert separator == ','  # Should fallback to comma
    
    def test_detect_csv_separator_error_handling(self):
        """Test separator detection error handling."""
        adapter = FileAdapter("/nonexistent/file.csv")
        
        separator = adapter._detect_csv_separator()
        
        assert separator == ','  # Should fallback to comma


class TestAutoDetection:
    """Test auto-detection functionality."""
    
    def test_auto_detect_csv(self, temp_dir, sample_csv_data):
        """Test auto-detection of CSV format."""
        # Create file without extension
        unknown_file = Path(temp_dir) / "unknown_file"
        sample_csv_data.to_csv(unknown_file, index=False)
        
        adapter = FileAdapter(str(unknown_file))
        result = adapter.load_data()
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_csv_data)
    
    def test_auto_detect_json(self, temp_dir, sample_json_data):
        """Test auto-detection of JSON format."""
        unknown_file = Path(temp_dir) / "unknown_file"
        
        with open(unknown_file, 'w') as f:
            json.dump(sample_json_data, f)
        
        adapter = FileAdapter(str(unknown_file))
        result = adapter.load_data()
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_json_data)
    
    def test_auto_detect_failure(self, temp_dir):
        """Test auto-detection failure handling."""
        unknown_file = Path(temp_dir) / "unknown_file"
        
        # Create file with unrecognizable format
        with open(unknown_file, 'wb') as f:
            f.write(b'\x00\x01\x02\x03\x04\x05')
        
        adapter = FileAdapter(str(unknown_file))
        
        with pytest.raises(Exception):
            adapter.load_data()


class TestFileInfo:
    """Test file information functionality."""
    
    def test_get_file_info(self, temp_dir, sample_csv_data):
        """Test getting file information."""
        csv_file = Path(temp_dir) / "test.csv"
        sample_csv_data.to_csv(csv_file, index=False)
        
        adapter = FileAdapter(str(csv_file))
        info = adapter.get_file_info()
        
        assert 'file_path' in info
        assert 'file_name' in info
        assert 'file_extension' in info
        assert 'file_size_bytes' in info
        assert 'file_size_mb' in info
        assert 'modified_time' in info
        
        assert info['file_name'] == 'test.csv'
        assert info['file_extension'] == '.csv'
        assert info['file_size_bytes'] > 0
        assert info['file_size_mb'] > 0
    
    def test_get_file_info_error_handling(self):
        """Test file info error handling."""
        adapter = FileAdapter("/nonexistent/file.csv")
        info = adapter.get_file_info()
        
        assert info == {}


class TestSampling:
    """Test data sampling functionality."""
    
    def test_sample_data_csv(self, temp_dir, sample_csv_data):
        """Test CSV data sampling."""
        csv_file = Path(temp_dir) / "test.csv"
        sample_csv_data.to_csv(csv_file, index=False)
        
        adapter = FileAdapter(str(csv_file))
        sample = adapter.sample_data(n_rows=3)
        
        assert isinstance(sample, pd.DataFrame)
        assert len(sample) == 3
        assert list(sample.columns) == list(sample_csv_data.columns)
    
    def test_sample_data_small_file(self, temp_dir):
        """Test sampling when file is smaller than requested sample."""
        csv_file = Path(temp_dir) / "small.csv"
        
        small_data = pd.DataFrame({'col1': [1, 2], 'col2': ['A', 'B']})
        small_data.to_csv(csv_file, index=False)
        
        adapter = FileAdapter(str(csv_file))
        sample = adapter.sample_data(n_rows=10)
        
        assert len(sample) == 2  # Should return all available data
    
    def test_sample_data_parquet(self, temp_dir, sample_csv_data):
        """Test Parquet data sampling."""
        parquet_file = Path(temp_dir) / "test.parquet"
        sample_csv_data.to_parquet(parquet_file, index=False)
        
        adapter = FileAdapter(str(parquet_file))
        sample = adapter.sample_data(n_rows=3)
        
        assert isinstance(sample, pd.DataFrame)
        assert len(sample) <= 3  # Might be less due to random sampling
    
    def test_sample_csv_efficient(self, temp_dir, sample_csv_data):
        """Test efficient CSV sampling."""
        csv_file = Path(temp_dir) / "test.csv"
        sample_csv_data.to_csv(csv_file, index=False)
        
        adapter = FileAdapter(str(csv_file))
        sample = adapter._sample_csv(3)
        
        assert isinstance(sample, pd.DataFrame)
        assert len(sample) == 3
    
    def test_sample_csv_fallback(self, temp_dir, sample_csv_data):
        """Test CSV sampling fallback when efficient method fails."""
        csv_file = Path(temp_dir) / "test.csv"
        sample_csv_data.to_csv(csv_file, index=False)
        
        adapter = FileAdapter(str(csv_file))
        
        # Mock detection to fail
        with patch.object(adapter, '_detect_encoding', side_effect=Exception("Error")):
            sample = adapter._sample_csv(3)
            
            assert isinstance(sample, pd.DataFrame)
            assert len(sample) <= 3


class TestMultiFileAdapter:
    """Test MultiFileAdapter class."""
    
    def test_multi_file_loading(self, temp_dir, sample_csv_data):
        """Test loading multiple files."""
        # Create multiple CSV files
        file1 = Path(temp_dir) / "file1.csv"
        file2 = Path(temp_dir) / "file2.csv"
        
        sample_csv_data.to_csv(file1, index=False)
        sample_csv_data.to_csv(file2, index=False)
        
        adapter = MultiFileAdapter([str(file1), str(file2)])
        result = adapter.load_data()
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_csv_data) * 2  # Combined data
        assert '_source_file' in result.columns
        assert 'file1.csv' in result['_source_file'].values
        assert 'file2.csv' in result['_source_file'].values
    
    def test_multi_file_loading_mixed_formats(self, temp_dir, sample_csv_data, sample_json_data):
        """Test loading multiple files with different formats."""
        csv_file = Path(temp_dir) / "data.csv"
        json_file = Path(temp_dir) / "data.json"
        
        sample_csv_data.to_csv(csv_file, index=False)
        
        with open(json_file, 'w') as f:
            json.dump(sample_json_data, f)
        
        adapter = MultiFileAdapter([str(csv_file), str(json_file)])
        result = adapter.load_data()
        
        assert isinstance(result, pd.DataFrame)
        assert '_source_file' in result.columns
    
    def test_multi_file_loading_with_errors(self, temp_dir, sample_csv_data):
        """Test multi-file loading with some files failing."""
        good_file = Path(temp_dir) / "good.csv"
        bad_file = Path(temp_dir) / "bad.csv"
        
        sample_csv_data.to_csv(good_file, index=False)
        
        # Create invalid CSV file
        with open(bad_file, 'w') as f:
            f.write("invalid,csv,content\n\x00\x01\x02")
        
        adapter = MultiFileAdapter([str(good_file), str(bad_file)])
        result = adapter.load_data()
        
        # Should load the good file and skip the bad one
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_csv_data)
        assert 'good.csv' in result['_source_file'].values
    
    def test_multi_file_loading_no_valid_files(self):
        """Test multi-file loading when no files can be loaded."""
        adapter = MultiFileAdapter(["/nonexistent1.csv", "/nonexistent2.csv"])
        
        with pytest.raises(ValueError, match="No files could be loaded successfully"):
            adapter.load_data()
    
    def test_get_files_info(self, temp_dir, sample_csv_data):
        """Test getting information about multiple files."""
        file1 = Path(temp_dir) / "file1.csv"
        file2 = Path(temp_dir) / "file2.csv"
        
        sample_csv_data.to_csv(file1, index=False)
        sample_csv_data.to_csv(file2, index=False)
        
        adapter = MultiFileAdapter([str(file1), str(file2)])
        files_info = adapter.get_files_info()
        
        assert len(files_info) == 2
        assert all('file_path' in info for info in files_info)
        assert all('file_name' in info for info in files_info)
    
    def test_get_files_info_with_errors(self, temp_dir, sample_csv_data):
        """Test getting files info with some files having errors."""
        good_file = Path(temp_dir) / "good.csv"
        sample_csv_data.to_csv(good_file, index=False)
        
        adapter = MultiFileAdapter([str(good_file), "/nonexistent.csv"])
        files_info = adapter.get_files_info()
        
        assert len(files_info) == 2
        assert files_info[0]['file_name'] == 'good.csv'
        assert 'error' in files_info[1]


class TestSpecializedFormatsWithMocks:
    """Test specialized formats with mocked dependencies."""
    
    def test_load_avro_success(self, temp_dir):
        """Test Avro loading with mocked fastavro."""
        avro_file = Path(temp_dir) / "test.avro"
        avro_file.touch()
        
        mock_records = [
            {'id': 1, 'name': 'Alice'},
            {'id': 2, 'name': 'Bob'}
        ]
        
        with patch('src.packages.data_profiling.infrastructure.adapters.file_adapter.fastavro') as mock_fastavro:
            mock_reader = mock_fastavro.reader.return_value
            mock_reader.__iter__ = lambda self: iter(mock_records)
            
            adapter = FileAdapter(str(avro_file))
            result = adapter.load_data()
            
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 2
            assert 'id' in result.columns
            assert 'name' in result.columns
    
    def test_load_avro_import_error(self, temp_dir):
        """Test Avro loading when fastavro is not available."""
        avro_file = Path(temp_dir) / "test.avro"
        avro_file.touch()
        
        adapter = FileAdapter(str(avro_file))
        
        with patch('src.packages.data_profiling.infrastructure.adapters.file_adapter.fastavro', side_effect=ImportError):
            with pytest.raises(ImportError):
                adapter.load_data()
    
    def test_load_orc_success(self, temp_dir):
        """Test ORC loading with mocked pyorc."""
        orc_file = Path(temp_dir) / "test.orc"
        orc_file.touch()
        
        mock_records = [
            {'id': 1, 'name': 'Alice'},
            {'id': 2, 'name': 'Bob'}
        ]
        
        with patch('src.packages.data_profiling.infrastructure.adapters.file_adapter.pyorc') as mock_pyorc:
            mock_reader = mock_pyorc.Reader.return_value
            mock_reader.__iter__ = lambda self: iter(mock_records)
            
            adapter = FileAdapter(str(orc_file))
            result = adapter.load_data()
            
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 2
    
    def test_load_orc_import_error(self, temp_dir):
        """Test ORC loading when pyorc is not available."""
        orc_file = Path(temp_dir) / "test.orc"
        orc_file.touch()
        
        adapter = FileAdapter(str(orc_file))
        
        with patch('src.packages.data_profiling.infrastructure.adapters.file_adapter.pyorc', side_effect=ImportError):
            with pytest.raises(ImportError):
                adapter.load_data()
    
    def test_load_hdf5_with_key_detection(self, temp_dir, sample_csv_data):
        """Test HDF5 loading with automatic key detection."""
        h5_file = Path(temp_dir) / "test.h5"
        
        # Create HDF5 file
        sample_csv_data.to_hdf(h5_file, key='data', mode='w')
        
        adapter = FileAdapter(str(h5_file))
        result = adapter.load_data()
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_csv_data)
    
    def test_load_hdf5_with_specified_key(self, temp_dir, sample_csv_data):
        """Test HDF5 loading with specified key."""
        h5_file = Path(temp_dir) / "test.h5"
        
        sample_csv_data.to_hdf(h5_file, key='specific_key', mode='w')
        
        adapter = FileAdapter(str(h5_file), key='specific_key')
        result = adapter.load_data()
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_csv_data)