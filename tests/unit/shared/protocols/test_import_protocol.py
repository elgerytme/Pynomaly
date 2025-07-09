"""Tests for import protocol."""

from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from pynomaly.domain.entities.dataset import Dataset
from pynomaly.shared.protocols.import_protocol import ImportProtocol


class TestImportProtocol:
    """Test suite for ImportProtocol."""

    def test_protocol_definition(self):
        """Test protocol has required abstract methods."""
        assert hasattr(ImportProtocol, 'import_dataset')
        assert hasattr(ImportProtocol, 'get_supported_formats')
        assert hasattr(ImportProtocol, 'validate_file')

    def test_abstract_base_class(self):
        """Test ImportProtocol is an abstract base class."""
        with pytest.raises(TypeError):
            ImportProtocol()

    def test_import_dataset_method_signature(self):
        """Test import_dataset method has correct signature."""
        class ConcreteImporter(ImportProtocol):
            def import_dataset(self, file_path: str | Path, options: dict[str, Any] = None) -> Dataset:
                return Mock(spec=Dataset)
            
            def get_supported_formats(self) -> list[str]:
                return [".csv"]
            
            def validate_file(self, file_path: str | Path) -> bool:
                return True
        
        importer = ConcreteImporter()
        mock_dataset = Mock(spec=Dataset)
        
        # Mock the method to return our mock dataset
        importer.import_dataset = Mock(return_value=mock_dataset)
        
        # Test with string file path
        result = importer.import_dataset("data.csv")
        assert result == mock_dataset
        
        # Test with Path object
        result = importer.import_dataset(Path("data.csv"))
        assert result == mock_dataset
        
        # Test with options
        result = importer.import_dataset("data.csv", options={"delimiter": ","})
        assert result == mock_dataset

    def test_get_supported_formats_method_signature(self):
        """Test get_supported_formats method has correct signature."""
        class ConcreteImporter(ImportProtocol):
            def import_dataset(self, file_path: str | Path, options: dict[str, Any] = None) -> Dataset:
                return Mock(spec=Dataset)
            
            def get_supported_formats(self) -> list[str]:
                return [".csv", ".xlsx", ".json"]
            
            def validate_file(self, file_path: str | Path) -> bool:
                return True
        
        importer = ConcreteImporter()
        formats = importer.get_supported_formats()
        
        assert isinstance(formats, list)
        assert all(isinstance(fmt, str) for fmt in formats)
        assert all(fmt.startswith('.') for fmt in formats)

    def test_validate_file_method_signature(self):
        """Test validate_file method has correct signature."""
        class ConcreteImporter(ImportProtocol):
            def import_dataset(self, file_path: str | Path, options: dict[str, Any] = None) -> Dataset:
                return Mock(spec=Dataset)
            
            def get_supported_formats(self) -> list[str]:
                return [".csv"]
            
            def validate_file(self, file_path: str | Path) -> bool:
                if isinstance(file_path, (str, Path)):
                    return str(file_path).endswith('.csv')
                return False
        
        importer = ConcreteImporter()
        
        # Test with string path
        assert importer.validate_file("data.csv") is True
        assert importer.validate_file("data.txt") is False
        
        # Test with Path object
        assert importer.validate_file(Path("data.csv")) is True
        assert importer.validate_file(Path("data.txt")) is False

    def test_concrete_csv_importer(self):
        """Test concrete CSV importer implementation."""
        class CSVImporter(ImportProtocol):
            def import_dataset(self, file_path: str | Path, options: dict[str, Any] = None) -> Dataset:
                mock_dataset = Mock(spec=Dataset)
                mock_dataset.name = f"dataset_from_{Path(file_path).name}"
                mock_dataset.source = str(file_path)
                return mock_dataset
            
            def get_supported_formats(self) -> list[str]:
                return [".csv"]
            
            def validate_file(self, file_path: str | Path) -> bool:
                return str(file_path).lower().endswith('.csv')
        
        importer = CSVImporter()
        
        # Test import
        dataset = importer.import_dataset("data.csv")
        assert hasattr(dataset, 'name')
        assert hasattr(dataset, 'source')
        
        # Test format support
        formats = importer.get_supported_formats()
        assert ".csv" in formats
        
        # Test validation
        assert importer.validate_file("data.csv") is True
        assert importer.validate_file("data.CSV") is True
        assert importer.validate_file("data.xlsx") is False

    def test_multi_format_importer(self):
        """Test importer supporting multiple formats."""
        class MultiFormatImporter(ImportProtocol):
            def import_dataset(self, file_path: str | Path, options: dict[str, Any] = None) -> Dataset:
                file_str = str(file_path)
                mock_dataset = Mock(spec=Dataset)
                
                if file_str.lower().endswith('.csv'):
                    mock_dataset.format = "csv"
                elif file_str.lower().endswith('.xlsx'):
                    mock_dataset.format = "excel"
                elif file_str.lower().endswith('.json'):
                    mock_dataset.format = "json"
                else:
                    raise ValueError(f"Unsupported format: {file_str}")
                
                return mock_dataset
            
            def get_supported_formats(self) -> list[str]:
                return [".csv", ".xlsx", ".json"]
            
            def validate_file(self, file_path: str | Path) -> bool:
                file_str = str(file_path).lower()
                return any(file_str.endswith(fmt.lower()) for fmt in self.get_supported_formats())
        
        importer = MultiFormatImporter()
        
        # Test different formats
        csv_dataset = importer.import_dataset("data.csv")
        assert csv_dataset.format == "csv"
        
        excel_dataset = importer.import_dataset("data.xlsx")
        assert excel_dataset.format == "excel"
        
        json_dataset = importer.import_dataset("data.json")
        assert json_dataset.format == "json"
        
        # Test validation
        assert importer.validate_file("data.csv") is True
        assert importer.validate_file("data.xlsx") is True
        assert importer.validate_file("data.json") is True
        assert importer.validate_file("data.txt") is False

    def test_importer_with_options(self):
        """Test importer that uses import options."""
        class ConfigurableImporter(ImportProtocol):
            def import_dataset(self, file_path: str | Path, options: dict[str, Any] = None) -> Dataset:
                mock_dataset = Mock(spec=Dataset)
                mock_dataset.source = str(file_path)
                
                if options:
                    mock_dataset.options = options
                    mock_dataset.delimiter = options.get("delimiter", ",")
                    mock_dataset.header = options.get("header", True)
                else:
                    mock_dataset.options = {}
                    mock_dataset.delimiter = ","
                    mock_dataset.header = True
                
                return mock_dataset
            
            def get_supported_formats(self) -> list[str]:
                return [".csv"]
            
            def validate_file(self, file_path: str | Path) -> bool:
                return True
        
        importer = ConfigurableImporter()
        
        # Test with options
        options = {"delimiter": ";", "header": False}
        dataset_with_options = importer.import_dataset("data.csv", options=options)
        assert dataset_with_options.options == options
        assert dataset_with_options.delimiter == ";"
        assert dataset_with_options.header is False
        
        # Test without options
        dataset_without_options = importer.import_dataset("data.csv")
        assert dataset_without_options.options == {}
        assert dataset_without_options.delimiter == ","
        assert dataset_without_options.header is True

    def test_importer_error_handling(self):
        """Test importer error handling."""
        class ErrorHandlingImporter(ImportProtocol):
            def import_dataset(self, file_path: str | Path, options: dict[str, Any] = None) -> Dataset:
                if str(file_path) == "invalid.txt":
                    raise ValueError("Unsupported file format")
                
                if str(file_path) == "missing.csv":
                    raise FileNotFoundError("File not found")
                
                return Mock(spec=Dataset)
            
            def get_supported_formats(self) -> list[str]:
                return [".csv"]
            
            def validate_file(self, file_path: str | Path) -> bool:
                return str(file_path).endswith('.csv')
        
        importer = ErrorHandlingImporter()
        
        # Test successful import
        dataset = importer.import_dataset("data.csv")
        assert isinstance(dataset, Mock)
        
        # Test error cases
        with pytest.raises(ValueError, match="Unsupported file format"):
            importer.import_dataset("invalid.txt")
        
        with pytest.raises(FileNotFoundError, match="File not found"):
            importer.import_dataset("missing.csv")

    def test_inheritance_behavior(self):
        """Test that subclasses must implement all abstract methods."""
        with pytest.raises(TypeError):
            class IncompleteImporter(ImportProtocol):
                def import_dataset(self, file_path: str | Path, options: dict[str, Any] = None) -> Dataset:
                    return Mock(spec=Dataset)
                # Missing get_supported_formats and validate_file
            
            IncompleteImporter()

    def test_method_contracts(self):
        """Test method contracts and return types."""
        class ContractTestImporter(ImportProtocol):
            def import_dataset(self, file_path: str | Path, options: dict[str, Any] = None) -> Dataset:
                return Mock(spec=Dataset)
            
            def get_supported_formats(self) -> list[str]:
                return [".csv", ".xlsx"]
            
            def validate_file(self, file_path: str | Path) -> bool:
                return True
        
        importer = ContractTestImporter()
        
        # Test return types
        dataset = importer.import_dataset("test.csv")
        assert isinstance(dataset, Mock)
        
        formats = importer.get_supported_formats()
        assert isinstance(formats, list)
        assert all(isinstance(fmt, str) for fmt in formats)
        
        is_valid = importer.validate_file("test.csv")
        assert isinstance(is_valid, bool)

    def test_path_handling(self):
        """Test handling of different path types."""
        class PathHandlingImporter(ImportProtocol):
            def import_dataset(self, file_path: str | Path, options: dict[str, Any] = None) -> Dataset:
                mock_dataset = Mock(spec=Dataset)
                mock_dataset.path_type = type(file_path).__name__
                mock_dataset.path_str = str(file_path)
                return mock_dataset
            
            def get_supported_formats(self) -> list[str]:
                return [".csv"]
            
            def validate_file(self, file_path: str | Path) -> bool:
                return str(file_path).endswith('.csv')
        
        importer = PathHandlingImporter()
        
        # Test with string path
        str_dataset = importer.import_dataset("data.csv")
        assert str_dataset.path_type == "str"
        assert str_dataset.path_str == "data.csv"
        
        # Test with Path object
        path_dataset = importer.import_dataset(Path("data.csv"))
        assert path_dataset.path_type == "PosixPath"
        assert str_dataset.path_str == "data.csv"

    def test_options_handling(self):
        """Test various options handling scenarios."""
        class OptionsImporter(ImportProtocol):
            def import_dataset(self, file_path: str | Path, options: dict[str, Any] = None) -> Dataset:
                mock_dataset = Mock(spec=Dataset)
                mock_dataset.has_options = options is not None
                mock_dataset.options_count = len(options) if options else 0
                return mock_dataset
            
            def get_supported_formats(self) -> list[str]:
                return [".csv"]
            
            def validate_file(self, file_path: str | Path) -> bool:
                return True
        
        importer = OptionsImporter()
        
        # Test with None options
        dataset_none = importer.import_dataset("data.csv", options=None)
        assert dataset_none.has_options is False
        assert dataset_none.options_count == 0
        
        # Test with empty options
        dataset_empty = importer.import_dataset("data.csv", options={})
        assert dataset_empty.has_options is True
        assert dataset_empty.options_count == 0
        
        # Test with options
        options = {"delimiter": ",", "header": True, "encoding": "utf-8"}
        dataset_with_options = importer.import_dataset("data.csv", options=options)
        assert dataset_with_options.has_options is True
        assert dataset_with_options.options_count == 3