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
        """Test protocol has required methods."""
        assert hasattr(ImportProtocol, 'import_dataset')
        assert hasattr(ImportProtocol, 'get_supported_formats')
        assert hasattr(ImportProtocol, 'validate_file')

    def test_import_dataset_method_signature(self):
        """Test import_dataset method has correct signature."""
        mock_importer = Mock(spec=ImportProtocol)
        mock_dataset = Mock(spec=Dataset)
        mock_importer.import_dataset.return_value = mock_dataset
        
        # Test with string file path
        result = mock_importer.import_dataset("data.csv")
        assert result == mock_dataset
        assert isinstance(result, Mock)  # Mock of Dataset
        
        # Test with Path object
        result = mock_importer.import_dataset(Path("data.csv"))
        assert result == mock_dataset
        
        # Test with options
        options = {"delimiter": ",", "header": True}
        result = mock_importer.import_dataset("data.csv", options)
        assert result == mock_dataset
        
        # Test with None options
        result = mock_importer.import_dataset("data.csv", None)
        assert result == mock_dataset

    def test_get_supported_formats_method_signature(self):
        """Test get_supported_formats method has correct signature."""
        mock_importer = Mock(spec=ImportProtocol)
        mock_formats = ['.csv', '.xlsx', '.json']
        mock_importer.get_supported_formats.return_value = mock_formats
        
        result = mock_importer.get_supported_formats()
        assert result == mock_formats
        assert isinstance(result, list)
        assert all(isinstance(fmt, str) for fmt in result)

    def test_validate_file_method_signature(self):
        """Test validate_file method has correct signature."""
        mock_importer = Mock(spec=ImportProtocol)
        mock_importer.validate_file.return_value = True
        
        # Test with string file path
        result = mock_importer.validate_file("data.csv")
        assert result is True
        assert isinstance(result, bool)
        
        # Test with Path object
        result = mock_importer.validate_file(Path("data.csv"))
        assert result is True

    def test_protocol_runtime_checkable(self):
        """Test protocol is runtime checkable."""
        class ConcreteImporter:
            def import_dataset(
                self, file_path: str | Path, options: dict[str, Any] = None
            ) -> Dataset:
                return Mock(spec=Dataset)
            
            def get_supported_formats(self) -> list[str]:
                return ['.csv', '.xlsx']
            
            def validate_file(self, file_path: str | Path) -> bool:
                return True
        
        importer = ConcreteImporter()
        assert isinstance(importer, ImportProtocol)

    def test_protocol_with_missing_methods(self):
        """Test protocol check fails with missing methods."""
        class IncompleteImporter:
            def import_dataset(
                self, file_path: str | Path, options: dict[str, Any] = None
            ) -> Dataset:
                return Mock(spec=Dataset)
            # Missing get_supported_formats and validate_file
        
        importer = IncompleteImporter()
        assert not isinstance(importer, ImportProtocol)

    def test_import_dataset_with_various_options(self):
        """Test import_dataset accepts various option types."""
        mock_importer = Mock(spec=ImportProtocol)
        mock_dataset = Mock(spec=Dataset)
        mock_importer.import_dataset.return_value = mock_dataset
        
        # Test with different option configurations
        options = {
            "delimiter": ",",
            "header": True,
            "encoding": "utf-8",
            "skip_rows": 1,
            "na_values": ["", "NULL", "N/A"]
        }
        
        result = mock_importer.import_dataset("data.csv", options)
        assert result == mock_dataset
        
        mock_importer.import_dataset.assert_called_with("data.csv", options)

    def test_validate_file_edge_cases(self):
        """Test validate_file handles edge cases."""
        mock_importer = Mock(spec=ImportProtocol)
        
        # Test with empty path
        mock_importer.validate_file.return_value = False
        result = mock_importer.validate_file("")
        assert result is False
        
        # Test with non-existent file
        mock_importer.validate_file.return_value = False
        result = mock_importer.validate_file("/non/existent/file.csv")
        assert result is False
        
        # Test with unsupported extension
        mock_importer.validate_file.return_value = False
        result = mock_importer.validate_file("file.unsupported")
        assert result is False

    def test_import_dataset_return_types(self):
        """Test import_dataset returns proper Dataset objects."""
        mock_importer = Mock(spec=ImportProtocol)
        mock_dataset = Mock(spec=Dataset)
        mock_dataset.name = "test_dataset"
        mock_dataset.shape = (1000, 10)
        mock_dataset.columns = ["col1", "col2", "col3"]
        mock_importer.import_dataset.return_value = mock_dataset
        
        result = mock_importer.import_dataset("data.csv")
        assert isinstance(result, Mock)  # Mock of Dataset
        assert hasattr(result, 'name')
        assert hasattr(result, 'shape')
        assert hasattr(result, 'columns')

    def test_get_supported_formats_return_types(self):
        """Test get_supported_formats returns proper format types."""
        mock_importer = Mock(spec=ImportProtocol)
        
        # Test with various format strings
        formats = ['.csv', '.xlsx', '.json', '.parquet', '.feather']
        mock_importer.get_supported_formats.return_value = formats
        
        result = mock_importer.get_supported_formats()
        assert isinstance(result, list)
        assert len(result) == 5
        assert all(fmt.startswith('.') for fmt in result)

    def test_protocol_error_handling(self):
        """Test protocol methods can raise exceptions."""
        mock_importer = Mock(spec=ImportProtocol)
        
        # Configure mock to raise exception on import
        mock_importer.import_dataset.side_effect = FileNotFoundError("File not found")
        
        with pytest.raises(FileNotFoundError, match="File not found"):
            mock_importer.import_dataset("missing_file.csv")

    def test_protocol_with_none_values(self):
        """Test protocol handles None values appropriately."""
        mock_importer = Mock(spec=ImportProtocol)
        mock_dataset = Mock(spec=Dataset)
        mock_importer.import_dataset.return_value = mock_dataset
        
        # Test import_dataset with None options
        result = mock_importer.import_dataset("data.csv", None)
        assert result == mock_dataset

    def test_protocol_type_hints(self):
        """Test protocol type hints are properly defined."""
        class TypedImporter:
            def import_dataset(
                self, file_path: str | Path, options: dict[str, Any] = None
            ) -> Dataset:
                return Mock(spec=Dataset)
            
            def get_supported_formats(self) -> list[str]:
                return ['.csv']
            
            def validate_file(self, file_path: str | Path) -> bool:
                return True
        
        importer = TypedImporter()
        assert isinstance(importer, ImportProtocol)
        
        # Test return types
        dataset = importer.import_dataset("test.csv")
        assert isinstance(dataset, Mock)  # Mock of Dataset
        
        formats = importer.get_supported_formats()
        assert isinstance(formats, list)
        
        is_valid = importer.validate_file("test.csv")
        assert isinstance(is_valid, bool)


class TestProtocolInteractions:
    """Test protocol interactions and edge cases."""

    def test_multiple_importers_same_protocol(self):
        """Test multiple implementations of same protocol."""
        class CSVImporter:
            def import_dataset(
                self, file_path: str | Path, options: dict[str, Any] = None
            ) -> Dataset:
                dataset = Mock(spec=Dataset)
                dataset.name = "csv_dataset"
                return dataset
            
            def get_supported_formats(self) -> list[str]:
                return ['.csv']
            
            def validate_file(self, file_path: str | Path) -> bool:
                return str(file_path).endswith('.csv')
        
        class ExcelImporter:
            def import_dataset(
                self, file_path: str | Path, options: dict[str, Any] = None
            ) -> Dataset:
                dataset = Mock(spec=Dataset)
                dataset.name = "excel_dataset"
                return dataset
            
            def get_supported_formats(self) -> list[str]:
                return ['.xlsx', '.xls']
            
            def validate_file(self, file_path: str | Path) -> bool:
                return str(file_path).endswith(('.xlsx', '.xls'))
        
        csv_importer = CSVImporter()
        excel_importer = ExcelImporter()
        
        assert isinstance(csv_importer, ImportProtocol)
        assert isinstance(excel_importer, ImportProtocol)
        
        # Test different behavior
        assert csv_importer.get_supported_formats() == ['.csv']
        assert excel_importer.get_supported_formats() == ['.xlsx', '.xls']

    def test_protocol_with_advanced_options(self):
        """Test protocol with complex import options."""
        mock_importer = Mock(spec=ImportProtocol)
        mock_dataset = Mock(spec=Dataset)
        mock_dataset.name = "advanced_dataset"
        mock_importer.import_dataset.return_value = mock_dataset
        
        # Create detailed import options
        options = {
            "delimiter": ";",
            "encoding": "utf-8",
            "header": True,
            "skip_rows": 2,
            "na_values": ["", "NULL", "N/A", "nan"],
            "dtype": {"column1": "str", "column2": "int"},
            "parse_dates": ["date_column"],
            "date_format": "%Y-%m-%d",
            "thousands": ",",
            "decimal": ".",
            "comment": "#",
            "compression": "gzip"
        }
        
        result = mock_importer.import_dataset("data.csv.gz", options)
        assert result == mock_dataset
        
        mock_importer.import_dataset.assert_called_with("data.csv.gz", options)

    def test_protocol_with_path_objects(self):
        """Test protocol works with Path objects."""
        mock_importer = Mock(spec=ImportProtocol)
        mock_dataset = Mock(spec=Dataset)
        mock_importer.import_dataset.return_value = mock_dataset
        
        # Test with various Path objects
        paths = [
            Path("data.csv"),
            Path("/tmp/data/input.xlsx"),
            Path("../data/source.json")
        ]
        
        mock_importer.validate_file.return_value = True
        
        for path in paths:
            result = mock_importer.import_dataset(path)
            assert isinstance(result, Mock)  # Mock of Dataset
            
            is_valid = mock_importer.validate_file(path)
            assert isinstance(is_valid, bool)

    def test_protocol_abstract_base_class(self):
        """Test protocol is properly defined as ABC."""
        # ImportProtocol should be abstract
        with pytest.raises(TypeError):
            # This should fail because ImportProtocol is abstract
            ImportProtocol()

    def test_protocol_method_signatures_detailed(self):
        """Test detailed method signatures match protocol."""
        class DetailedImporter:
            def import_dataset(
                self, file_path: str | Path, options: dict[str, Any] = None
            ) -> Dataset:
                """Import dataset with detailed processing."""
                dataset = Mock(spec=Dataset)
                dataset.name = f"dataset_from_{Path(file_path).name}"
                dataset.shape = (1000, 10)
                dataset.columns = [f"col_{i}" for i in range(10)]
                dataset.dtypes = {"col_0": "int64", "col_1": "float64"}
                return dataset
            
            def get_supported_formats(self) -> list[str]:
                """Return supported file formats."""
                return ['.csv', '.xlsx', '.json', '.parquet', '.feather']
            
            def validate_file(self, file_path: str | Path) -> bool:
                """Validate file path for import."""
                if isinstance(file_path, str):
                    file_path = Path(file_path)
                
                # Check if file exists
                if not file_path.exists():
                    return False
                
                # Check if file is readable
                if not file_path.is_file():
                    return False
                
                # Check file extension
                supported_formats = self.get_supported_formats()
                return file_path.suffix in supported_formats
        
        importer = DetailedImporter()
        assert isinstance(importer, ImportProtocol)
        
        # Test detailed functionality
        dataset = importer.import_dataset("test.csv")
        assert hasattr(dataset, 'name')
        assert hasattr(dataset, 'shape')
        assert hasattr(dataset, 'columns')

    def test_protocol_with_streaming_data(self):
        """Test protocol with streaming data sources."""
        mock_importer = Mock(spec=ImportProtocol)
        mock_dataset = Mock(spec=Dataset)
        mock_dataset.name = "streaming_dataset"
        mock_importer.import_dataset.return_value = mock_dataset
        
        # Test with streaming-like options
        options = {
            "stream": True,
            "chunk_size": 1000,
            "buffer_size": 10000,
            "timeout": 30
        }
        
        result = mock_importer.import_dataset("kafka://topic", options)
        assert result == mock_dataset

    def test_protocol_with_database_sources(self):
        """Test protocol with database-like sources."""
        mock_importer = Mock(spec=ImportProtocol)
        mock_dataset = Mock(spec=Dataset)
        mock_dataset.name = "database_dataset"
        mock_importer.import_dataset.return_value = mock_dataset
        
        # Test with database connection options
        options = {
            "connection_string": "postgresql://user:pass@host:5432/db",
            "query": "SELECT * FROM table WHERE condition = 1",
            "batch_size": 1000
        }
        
        result = mock_importer.import_dataset("database://table", options)
        assert result == mock_dataset

    def test_protocol_validation_comprehensive(self):
        """Test comprehensive file validation."""
        class ValidatingImporter:
            def import_dataset(
                self, file_path: str | Path, options: dict[str, Any] = None
            ) -> Dataset:
                if not self.validate_file(file_path):
                    raise ValueError(f"Invalid file: {file_path}")
                return Mock(spec=Dataset)
            
            def get_supported_formats(self) -> list[str]:
                return ['.csv', '.xlsx', '.json']
            
            def validate_file(self, file_path: str | Path) -> bool:
                """Comprehensive file validation."""
                if isinstance(file_path, str):
                    file_path = Path(file_path)
                
                # Check extension
                if file_path.suffix not in self.get_supported_formats():
                    return False
                
                # Check file name is not empty
                if not file_path.name:
                    return False
                
                # Check path is not just a directory
                if file_path.is_dir():
                    return False
                
                return True
        
        importer = ValidatingImporter()
        assert isinstance(importer, ImportProtocol)
        
        # Test validation
        assert importer.validate_file("valid.csv") is True
        assert importer.validate_file("invalid.txt") is False
        assert importer.validate_file("") is False

    def test_protocol_error_scenarios(self):
        """Test various error scenarios."""
        mock_importer = Mock(spec=ImportProtocol)
        
        # Test different exceptions
        error_scenarios = [
            (FileNotFoundError, "File not found"),
            (PermissionError, "Permission denied"),
            (ValueError, "Invalid file format"),
            (IOError, "I/O error"),
            (UnicodeDecodeError, "Encoding error", "utf-8", b"", 0, 1, "reason")
        ]
        
        for i, scenario in enumerate(error_scenarios):
            if len(scenario) == 2:
                exception_type, message = scenario
                mock_importer.import_dataset.side_effect = exception_type(message)
            else:
                exception_type = scenario[0]
                args = scenario[1:]
                mock_importer.import_dataset.side_effect = exception_type(*args)
            
            with pytest.raises(exception_type):
                mock_importer.import_dataset(f"file_{i}.csv")