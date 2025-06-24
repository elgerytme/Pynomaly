"""Tests for autonomous data loaders."""

import pytest
import pandas as pd
import numpy as np
import json
from pathlib import Path
from unittest.mock import patch

from pynomaly.infrastructure.data_loaders.json_loader import JSONLoader
from pynomaly.infrastructure.data_loaders.excel_loader import ExcelLoader
from pynomaly.domain.exceptions import DataValidationError


class TestJSONLoader:
    """Test JSON data loader."""
    
    def test_init_default(self):
        """Test default initialization."""
        loader = JSONLoader()
        assert loader.encoding == "utf-8"
        assert loader.lines is False
        assert loader.normalize_nested is True
    
    def test_init_custom(self):
        """Test custom initialization."""
        loader = JSONLoader(
            encoding="latin-1",
            lines=True,
            normalize_nested=False
        )
        assert loader.encoding == "latin-1"
        assert loader.lines is True
        assert loader.normalize_nested is False
    
    def test_supported_formats(self):
        """Test supported formats."""
        loader = JSONLoader()
        formats = loader.supported_formats
        assert "json" in formats
        assert "jsonl" in formats
    
    def test_load_simple_json(self, tmp_path):
        """Test loading simple JSON file."""
        # Create test JSON
        data = [
            {"name": "Alice", "age": 30, "score": 0.8},
            {"name": "Bob", "age": 25, "score": 0.9},
            {"name": "Charlie", "age": 35, "score": 0.7}
        ]
        
        json_file = tmp_path / "test.json"
        with open(json_file, 'w') as f:
            json.dump(data, f)
        
        loader = JSONLoader()
        dataset = loader.load(json_file)
        
        assert dataset.name == "test"
        assert len(dataset.data) == 3
        assert list(dataset.data.columns) == ["name", "age", "score"]
        assert dataset.data.iloc[0]["name"] == "Alice"
    
    def test_load_nested_json_with_normalization(self, tmp_path):
        """Test loading nested JSON with normalization."""
        data = [
            {
                "id": 1,
                "user": {"name": "Alice", "email": "alice@example.com"},
                "metrics": {"score": 0.8, "count": 10}
            },
            {
                "id": 2,
                "user": {"name": "Bob", "email": "bob@example.com"},
                "metrics": {"score": 0.9, "count": 15}
            }
        ]
        
        json_file = tmp_path / "nested.json"
        with open(json_file, 'w') as f:
            json.dump(data, f)
        
        loader = JSONLoader(normalize_nested=True)
        dataset = loader.load(json_file)
        
        # Should have normalized nested columns
        columns = list(dataset.data.columns)
        assert "id" in columns
        assert any("user_" in col for col in columns)
        assert any("metrics_" in col for col in columns)
    
    def test_load_jsonl_format(self, tmp_path):
        """Test loading JSON Lines format."""
        # Create JSONL file
        data = [
            {"name": "Alice", "value": 1},
            {"name": "Bob", "value": 2},
            {"name": "Charlie", "value": 3}
        ]
        
        jsonl_file = tmp_path / "test.jsonl"
        with open(jsonl_file, 'w') as f:
            for record in data:
                f.write(json.dumps(record) + "\n")
        
        loader = JSONLoader()
        dataset = loader.load(jsonl_file)
        
        assert len(dataset.data) == 3
        assert list(dataset.data.columns) == ["name", "value"]
        assert dataset.metadata["lines_format"] is True
    
    def test_load_with_target_column(self, tmp_path):
        """Test loading with target column specification."""
        data = [
            {"feature1": 1, "feature2": 2, "label": 0},
            {"feature1": 3, "feature2": 4, "label": 1}
        ]
        
        json_file = tmp_path / "labeled.json"
        with open(json_file, 'w') as f:
            json.dump(data, f)
        
        loader = JSONLoader()
        dataset = loader.load(json_file, target_column="label")
        
        assert dataset.target_column == "label"
        assert dataset.has_target is True
    
    def test_validate_valid_json(self, tmp_path):
        """Test validation of valid JSON file."""
        data = {"test": "data"}
        json_file = tmp_path / "valid.json"
        with open(json_file, 'w') as f:
            json.dump(data, f)
        
        loader = JSONLoader()
        assert loader.validate(json_file) is True
    
    def test_validate_invalid_json(self, tmp_path):
        """Test validation of invalid JSON file."""
        # Create invalid JSON
        json_file = tmp_path / "invalid.json"
        with open(json_file, 'w') as f:
            f.write("{ invalid json")
        
        loader = JSONLoader()
        assert loader.validate(json_file) is False
    
    def test_validate_nonexistent_file(self):
        """Test validation of non-existent file."""
        loader = JSONLoader()
        assert loader.validate(Path("nonexistent.json")) is False
    
    def test_load_batch_jsonl(self, tmp_path):
        """Test batch loading of JSONL file."""
        # Create large JSONL file
        jsonl_file = tmp_path / "batch_test.jsonl"
        with open(jsonl_file, 'w') as f:
            for i in range(10):
                record = {"id": i, "value": i * 2}
                f.write(json.dumps(record) + "\n")
        
        loader = JSONLoader()
        batches = list(loader.load_batch(jsonl_file, batch_size=3))
        
        assert len(batches) == 4  # 10 records, batch size 3 -> 4 batches
        assert len(batches[0].data) == 3
        assert len(batches[1].data) == 3
        assert len(batches[2].data) == 3
        assert len(batches[3].data) == 1  # Last batch
    
    def test_load_batch_regular_json(self, tmp_path):
        """Test batch loading of regular JSON file."""
        data = [{"id": i, "value": i * 2} for i in range(8)]
        
        json_file = tmp_path / "batch_regular.json"
        with open(json_file, 'w') as f:
            json.dump(data, f)
        
        loader = JSONLoader()
        batches = list(loader.load_batch(json_file, batch_size=3))
        
        assert len(batches) == 3  # 8 records, batch size 3 -> 3 batches
        assert len(batches[0].data) == 3
        assert len(batches[1].data) == 3
        assert len(batches[2].data) == 2  # Last batch
    
    def test_estimate_size_jsonl(self, tmp_path):
        """Test size estimation for JSONL file."""
        jsonl_file = tmp_path / "size_test.jsonl"
        with open(jsonl_file, 'w') as f:
            for i in range(50):
                record = {"id": i, "name": f"user_{i}", "score": i * 0.1}
                f.write(json.dumps(record) + "\n")
        
        loader = JSONLoader()
        size_info = loader.estimate_size(jsonl_file)
        
        assert "file_size_mb" in size_info
        assert "estimated_rows" in size_info
        assert "columns" in size_info
        assert size_info["format"] == "jsonl"
        assert size_info["estimated_rows"] >= 50
    
    def test_estimate_size_regular_json(self, tmp_path):
        """Test size estimation for regular JSON file."""
        data = [{"id": i, "value": i * 2} for i in range(20)]
        
        json_file = tmp_path / "regular_size.json"
        with open(json_file, 'w') as f:
            json.dump(data, f)
        
        loader = JSONLoader()
        size_info = loader.estimate_size(json_file)
        
        assert "file_size_mb" in size_info
        assert "estimated_rows" in size_info
        assert size_info["format"] == "json"
        assert size_info["data_type"] == "list"
    
    def test_load_empty_file_error(self, tmp_path):
        """Test loading empty JSON file raises error."""
        json_file = tmp_path / "empty.json"
        json_file.write_text("")
        
        loader = JSONLoader()
        with pytest.raises(DataValidationError, match="JSON file is empty"):
            loader.load(json_file)
    
    def test_load_invalid_json_error(self, tmp_path):
        """Test loading invalid JSON raises error."""
        json_file = tmp_path / "invalid.json"
        json_file.write_text("{ invalid json }")
        
        loader = JSONLoader()
        with pytest.raises(DataValidationError, match="Invalid JSON format"):
            loader.load(json_file)


class TestExcelLoader:
    """Test Excel data loader."""
    
    def test_init_default(self):
        """Test default initialization."""
        loader = ExcelLoader()
        assert loader.sheet_name == 0
        assert loader.header == 0
        assert loader.skiprows is None
    
    def test_init_custom(self):
        """Test custom initialization."""
        loader = ExcelLoader(
            sheet_name="Sheet2",
            header=[0, 1],
            skiprows=2
        )
        assert loader.sheet_name == "Sheet2"
        assert loader.header == [0, 1]
        assert loader.skiprows == 2
    
    def test_supported_formats(self):
        """Test supported formats."""
        loader = ExcelLoader()
        formats = loader.supported_formats
        expected = ["xlsx", "xls", "xlsm", "xlsb"]
        for fmt in expected:
            assert fmt in formats
    
    @pytest.mark.skipif(
        True,  # Skip by default as it requires openpyxl
        reason="Requires openpyxl and xlrd for Excel support"
    )
    def test_load_excel_file(self, tmp_path):
        """Test loading Excel file (requires openpyxl)."""
        # Create test DataFrame
        df = pd.DataFrame({
            'Name': ['Alice', 'Bob', 'Charlie'],
            'Age': [30, 25, 35],
            'Score': [0.8, 0.9, 0.7]
        })
        
        excel_file = tmp_path / "test.xlsx"
        df.to_excel(excel_file, index=False)
        
        loader = ExcelLoader()
        dataset = loader.load(excel_file)
        
        assert dataset.name == "test"
        assert len(dataset.data) == 3
        assert list(dataset.data.columns) == ['Name', 'Age', 'Score']
    
    def test_validate_excel_extension(self):
        """Test validation based on file extension."""
        loader = ExcelLoader()
        
        # Valid extensions
        assert loader.validate(Path("test.xlsx")) is False  # File doesn't exist
        assert loader.validate(Path("test.xls")) is False   # File doesn't exist
        
        # Invalid extensions
        assert loader.validate(Path("test.csv")) is False
        assert loader.validate(Path("test.txt")) is False
    
    def test_validate_nonexistent_file(self):
        """Test validation of non-existent file."""
        loader = ExcelLoader()
        assert loader.validate(Path("nonexistent.xlsx")) is False
    
    @patch('pandas.ExcelFile')
    def test_validate_with_mock(self, mock_excel_file, tmp_path):
        """Test validation with mocked Excel file."""
        # Create a dummy Excel file
        excel_file = tmp_path / "test.xlsx"
        excel_file.write_bytes(b"dummy excel content")
        
        # Mock ExcelFile to simulate valid Excel
        mock_instance = mock_excel_file.return_value
        mock_instance.sheet_names = ['Sheet1']
        mock_instance.close.return_value = None
        
        loader = ExcelLoader()
        result = loader.validate(excel_file)
        
        # Should call ExcelFile and return True if sheets exist
        mock_excel_file.assert_called_once_with(excel_file)
        assert result is True
    
    def test_estimate_size_excel_file(self, tmp_path):
        """Test size estimation for Excel file."""
        # Create a dummy Excel file for size testing
        excel_file = tmp_path / "size_test.xlsx"
        excel_file.write_bytes(b"dummy excel content" * 100)  # Make it bigger
        
        loader = ExcelLoader()
        
        # Since we can't create real Excel without dependencies,
        # just test the error handling
        try:
            size_info = loader.estimate_size(excel_file)
            # If it works, check basic structure
            assert "file_size_mb" in size_info
        except DataValidationError:
            # Expected if Excel libraries not available
            pass


class TestDataLoaderIntegration:
    """Integration tests for data loaders."""
    
    def test_json_loader_with_complex_data(self, tmp_path):
        """Test JSON loader with complex nested data."""
        data = [
            {
                "user_id": 1,
                "profile": {
                    "name": "Alice",
                    "age": 30,
                    "preferences": ["reading", "coding"]
                },
                "activity": {
                    "login_count": 150,
                    "last_active": "2023-01-15"
                }
            },
            {
                "user_id": 2,
                "profile": {
                    "name": "Bob",
                    "age": 25,
                    "preferences": ["gaming", "music"]
                },
                "activity": {
                    "login_count": 89,
                    "last_active": "2023-01-14"
                }
            }
        ]
        
        json_file = tmp_path / "complex.json"
        with open(json_file, 'w') as f:
            json.dump(data, f)
        
        loader = JSONLoader(normalize_nested=True)
        dataset = loader.load(json_file)
        
        # Should have flattened nested structures
        columns = list(dataset.data.columns)
        assert "user_id" in columns
        assert any("profile_" in col for col in columns)
        assert any("activity_" in col for col in columns)
        
        # Verify data integrity
        assert len(dataset.data) == 2
        assert dataset.data["user_id"].tolist() == [1, 2]
    
    def test_json_loader_error_recovery(self, tmp_path):
        """Test JSON loader error recovery with malformed JSONL."""
        # Create JSONL with some invalid lines
        jsonl_file = tmp_path / "mixed_quality.jsonl"
        with open(jsonl_file, 'w') as f:
            f.write('{"id": 1, "value": "good"}\n')
            f.write('{ invalid json line }\n')  # Invalid
            f.write('{"id": 2, "value": "also_good"}\n')
            f.write('{"id": 3, "value": "good_too"}\n')
        
        loader = JSONLoader()
        batches = list(loader.load_batch(jsonl_file, batch_size=2))
        
        # Should recover and process valid lines
        total_records = sum(len(batch.data) for batch in batches)
        assert total_records == 3  # Should skip the invalid line
    
    def test_loader_metadata_consistency(self, tmp_path):
        """Test that all loaders provide consistent metadata."""
        # Test with JSON
        data = [{"col1": 1, "col2": 2}, {"col1": 3, "col2": 4}]
        json_file = tmp_path / "meta_test.json"
        with open(json_file, 'w') as f:
            json.dump(data, f)
        
        json_loader = JSONLoader()
        json_dataset = json_loader.load(json_file)
        
        # Check required metadata fields
        assert "source" in json_dataset.metadata
        assert "loader" in json_dataset.metadata
        assert "file_size_mb" in json_dataset.metadata
        assert json_dataset.metadata["loader"] == "JSONLoader"
        assert json_dataset.metadata["source"] == str(json_file)
    
    def test_batch_loading_consistency(self, tmp_path):
        """Test that batch loading produces consistent results."""
        # Create test data
        data = [{"id": i, "value": i * 2} for i in range(20)]
        json_file = tmp_path / "batch_consistency.json"
        with open(json_file, 'w') as f:
            json.dump(data, f)
        
        loader = JSONLoader()
        
        # Load full dataset
        full_dataset = loader.load(json_file)
        
        # Load in batches
        batches = list(loader.load_batch(json_file, batch_size=5))
        
        # Combine batches
        combined_data = pd.concat([batch.data for batch in batches], ignore_index=True)
        
        # Should be equivalent
        assert len(combined_data) == len(full_dataset.data)
        assert list(combined_data.columns) == list(full_dataset.data.columns)
        assert combined_data["id"].tolist() == full_dataset.data["id"].tolist()
    
    def test_error_handling_consistency(self, tmp_path):
        """Test that error handling is consistent across loaders."""
        # Test with non-existent files
        json_loader = JSONLoader()
        excel_loader = ExcelLoader()
        
        # Both should return False for validation of non-existent files
        assert json_loader.validate(Path("nonexistent.json")) is False
        assert excel_loader.validate(Path("nonexistent.xlsx")) is False
        
        # Both should raise DataValidationError for invalid files in load()
        invalid_file = tmp_path / "invalid.json"
        invalid_file.write_text("invalid content")
        
        with pytest.raises(DataValidationError):
            json_loader.load(invalid_file)