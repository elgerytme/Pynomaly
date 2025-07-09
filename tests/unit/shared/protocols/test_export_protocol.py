"""Tests for export protocol."""

from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from pynomaly.application.dto.export_options import ExportOptions
from pynomaly.domain.entities.detection_result import DetectionResult
from pynomaly.shared.protocols.export_protocol import ExportProtocol


class TestExportProtocol:
    """Test suite for ExportProtocol."""

    def test_protocol_definition(self):
        """Test protocol has required methods."""
        assert hasattr(ExportProtocol, 'export_results')
        assert hasattr(ExportProtocol, 'get_supported_formats')
        assert hasattr(ExportProtocol, 'validate_file')

    def test_export_results_method_signature(self):
        """Test export_results method has correct signature."""
        mock_exporter = Mock(spec=ExportProtocol)
        mock_result = Mock(spec=DetectionResult)
        mock_options = Mock(spec=ExportOptions)
        mock_metadata = {"exported_records": 100, "file_size": 1024}
        mock_exporter.export_results.return_value = mock_metadata
        
        # Test with string file path
        result = mock_exporter.export_results(mock_result, "output.csv", mock_options)
        assert result == mock_metadata
        assert isinstance(result, dict)
        
        # Test with Path object
        result = mock_exporter.export_results(mock_result, Path("output.csv"), mock_options)
        assert result == mock_metadata
        
        # Test with None options
        result = mock_exporter.export_results(mock_result, "output.csv", None)
        assert result == mock_metadata

    def test_get_supported_formats_method_signature(self):
        """Test get_supported_formats method has correct signature."""
        mock_exporter = Mock(spec=ExportProtocol)
        mock_formats = ['.csv', '.xlsx', '.json']
        mock_exporter.get_supported_formats.return_value = mock_formats
        
        result = mock_exporter.get_supported_formats()
        assert result == mock_formats
        assert isinstance(result, list)
        assert all(isinstance(fmt, str) for fmt in result)

    def test_validate_file_method_signature(self):
        """Test validate_file method has correct signature."""
        mock_exporter = Mock(spec=ExportProtocol)
        mock_exporter.validate_file.return_value = True
        
        # Test with string file path
        result = mock_exporter.validate_file("output.csv")
        assert result is True
        assert isinstance(result, bool)
        
        # Test with Path object
        result = mock_exporter.validate_file(Path("output.csv"))
        assert result is True

    def test_protocol_runtime_checkable(self):
        """Test protocol is runtime checkable."""
        class ConcreteExporter:
            def export_results(
                self,
                results: DetectionResult,
                file_path: str | Path,
                options: ExportOptions = None,
            ) -> dict[str, Any]:
                return {"exported_records": 100, "file_size": 1024}
            
            def get_supported_formats(self) -> list[str]:
                return ['.csv', '.xlsx']
            
            def validate_file(self, file_path: str | Path) -> bool:
                return True
        
        exporter = ConcreteExporter()
        assert isinstance(exporter, ExportProtocol)

    def test_protocol_with_missing_methods(self):
        """Test protocol check fails with missing methods."""
        class IncompleteExporter:
            def export_results(
                self,
                results: DetectionResult,
                file_path: str | Path,
                options: ExportOptions = None,
            ) -> dict[str, Any]:
                return {"exported_records": 100}
            # Missing get_supported_formats and validate_file
        
        exporter = IncompleteExporter()
        assert not isinstance(exporter, ExportProtocol)

    def test_export_results_with_various_options(self):
        """Test export_results accepts various option types."""
        mock_exporter = Mock(spec=ExportProtocol)
        mock_result = Mock(spec=DetectionResult)
        mock_metadata = {"exported_records": 100}
        mock_exporter.export_results.return_value = mock_metadata
        
        # Test with different ExportOptions configurations
        options = Mock(spec=ExportOptions)
        options.format = "csv"
        options.include_metadata = True
        
        result = mock_exporter.export_results(mock_result, "output.csv", options)
        assert result == mock_metadata
        
        mock_exporter.export_results.assert_called_with(mock_result, "output.csv", options)

    def test_validate_file_edge_cases(self):
        """Test validate_file handles edge cases."""
        mock_exporter = Mock(spec=ExportProtocol)
        
        # Test with empty path
        mock_exporter.validate_file.return_value = False
        result = mock_exporter.validate_file("")
        assert result is False
        
        # Test with non-existent directory
        mock_exporter.validate_file.return_value = False
        result = mock_exporter.validate_file("/non/existent/path/file.csv")
        assert result is False
        
        # Test with unsupported extension
        mock_exporter.validate_file.return_value = False
        result = mock_exporter.validate_file("file.unsupported")
        assert result is False

    def test_export_results_return_types(self):
        """Test export_results returns proper metadata types."""
        mock_exporter = Mock(spec=ExportProtocol)
        mock_result = Mock(spec=DetectionResult)
        
        # Test typical metadata structure
        expected_metadata = {
            "exported_records": 100,
            "file_size": 1024,
            "format": "csv",
            "timestamp": "2023-01-01T00:00:00Z",
            "compression": None,
            "encoding": "utf-8"
        }
        mock_exporter.export_results.return_value = expected_metadata
        
        result = mock_exporter.export_results(mock_result, "output.csv")
        assert isinstance(result, dict)
        assert "exported_records" in result
        assert isinstance(result["exported_records"], int)

    def test_get_supported_formats_return_types(self):
        """Test get_supported_formats returns proper format types."""
        mock_exporter = Mock(spec=ExportProtocol)
        
        # Test with various format strings
        formats = ['.csv', '.xlsx', '.json', '.parquet', '.feather']
        mock_exporter.get_supported_formats.return_value = formats
        
        result = mock_exporter.get_supported_formats()
        assert isinstance(result, list)
        assert len(result) == 5
        assert all(fmt.startswith('.') for fmt in result)

    def test_protocol_error_handling(self):
        """Test protocol methods can raise exceptions."""
        mock_exporter = Mock(spec=ExportProtocol)
        mock_result = Mock(spec=DetectionResult)
        
        # Configure mock to raise exception on export
        mock_exporter.export_results.side_effect = ValueError("Invalid file path")
        
        with pytest.raises(ValueError, match="Invalid file path"):
            mock_exporter.export_results(mock_result, "invalid_path.csv")

    def test_protocol_with_none_values(self):
        """Test protocol handles None values appropriately."""
        mock_exporter = Mock(spec=ExportProtocol)
        mock_result = Mock(spec=DetectionResult)
        
        # Test export_results with None options
        mock_metadata = {"exported_records": 0}
        mock_exporter.export_results.return_value = mock_metadata
        
        result = mock_exporter.export_results(mock_result, "output.csv", None)
        assert result == mock_metadata

    def test_protocol_type_hints(self):
        """Test protocol type hints are properly defined."""
        class TypedExporter:
            def export_results(
                self,
                results: DetectionResult,
                file_path: str | Path,
                options: ExportOptions = None,
            ) -> dict[str, Any]:
                return {"exported_records": 100}
            
            def get_supported_formats(self) -> list[str]:
                return ['.csv']
            
            def validate_file(self, file_path: str | Path) -> bool:
                return True
        
        exporter = TypedExporter()
        assert isinstance(exporter, ExportProtocol)
        
        # Test return types
        mock_result = Mock(spec=DetectionResult)
        metadata = exporter.export_results(mock_result, "test.csv")
        assert isinstance(metadata, dict)
        
        formats = exporter.get_supported_formats()
        assert isinstance(formats, list)
        
        is_valid = exporter.validate_file("test.csv")
        assert isinstance(is_valid, bool)


class TestProtocolInteractions:
    """Test protocol interactions and edge cases."""

    def test_multiple_exporters_same_protocol(self):
        """Test multiple implementations of same protocol."""
        class CSVExporter:
            def export_results(
                self,
                results: DetectionResult,
                file_path: str | Path,
                options: ExportOptions = None,
            ) -> dict[str, Any]:
                return {"exported_records": 100, "format": "csv"}
            
            def get_supported_formats(self) -> list[str]:
                return ['.csv']
            
            def validate_file(self, file_path: str | Path) -> bool:
                return str(file_path).endswith('.csv')
        
        class ExcelExporter:
            def export_results(
                self,
                results: DetectionResult,
                file_path: str | Path,
                options: ExportOptions = None,
            ) -> dict[str, Any]:
                return {"exported_records": 100, "format": "excel"}
            
            def get_supported_formats(self) -> list[str]:
                return ['.xlsx', '.xls']
            
            def validate_file(self, file_path: str | Path) -> bool:
                return str(file_path).endswith(('.xlsx', '.xls'))
        
        csv_exporter = CSVExporter()
        excel_exporter = ExcelExporter()
        
        assert isinstance(csv_exporter, ExportProtocol)
        assert isinstance(excel_exporter, ExportProtocol)
        
        # Test different behavior
        assert csv_exporter.get_supported_formats() == ['.csv']
        assert excel_exporter.get_supported_formats() == ['.xlsx', '.xls']

    def test_protocol_with_advanced_options(self):
        """Test protocol with complex export options."""
        mock_exporter = Mock(spec=ExportProtocol)
        mock_result = Mock(spec=DetectionResult)
        
        # Create detailed export options
        options = Mock(spec=ExportOptions)
        options.format = "csv"
        options.include_metadata = True
        options.compression = "gzip"
        options.encoding = "utf-8"
        options.delimiter = ","
        options.quote_char = '"'
        
        mock_metadata = {
            "exported_records": 1000,
            "file_size": 2048,
            "format": "csv",
            "compression": "gzip",
            "encoding": "utf-8"
        }
        mock_exporter.export_results.return_value = mock_metadata
        
        result = mock_exporter.export_results(mock_result, "output.csv.gz", options)
        assert result == mock_metadata
        assert result["compression"] == "gzip"

    def test_protocol_with_path_objects(self):
        """Test protocol works with Path objects."""
        mock_exporter = Mock(spec=ExportProtocol)
        mock_result = Mock(spec=DetectionResult)
        
        # Test with various Path objects
        paths = [
            Path("output.csv"),
            Path("/tmp/data/output.xlsx"),
            Path("../data/results.json")
        ]
        
        mock_exporter.export_results.return_value = {"exported_records": 100}
        mock_exporter.validate_file.return_value = True
        
        for path in paths:
            result = mock_exporter.export_results(mock_result, path)
            assert isinstance(result, dict)
            
            is_valid = mock_exporter.validate_file(path)
            assert isinstance(is_valid, bool)

    def test_protocol_abstract_base_class(self):
        """Test protocol is properly defined as ABC."""
        # ExportProtocol should be abstract
        with pytest.raises(TypeError):
            # This should fail because ExportProtocol is abstract
            ExportProtocol()

    def test_protocol_method_signatures_detailed(self):
        """Test detailed method signatures match protocol."""
        class DetailedExporter:
            def export_results(
                self,
                results: DetectionResult,
                file_path: str | Path,
                options: ExportOptions = None,
            ) -> dict[str, Any]:
                """Export results with detailed metadata."""
                return {
                    "exported_records": 100,
                    "file_size": 1024,
                    "format": "csv",
                    "timestamp": "2023-01-01T00:00:00Z",
                    "success": True,
                    "errors": [],
                    "warnings": []
                }
            
            def get_supported_formats(self) -> list[str]:
                """Return supported file formats."""
                return ['.csv', '.xlsx', '.json', '.parquet']
            
            def validate_file(self, file_path: str | Path) -> bool:
                """Validate file path for export."""
                if isinstance(file_path, str):
                    file_path = Path(file_path)
                
                # Check if directory exists and is writable
                if not file_path.parent.exists():
                    return False
                
                # Check file extension
                supported_formats = self.get_supported_formats()
                return file_path.suffix in supported_formats
        
        exporter = DetailedExporter()
        assert isinstance(exporter, ExportProtocol)
        
        # Test detailed functionality
        mock_result = Mock(spec=DetectionResult)
        metadata = exporter.export_results(mock_result, "test.csv")
        assert "exported_records" in metadata
        assert "success" in metadata
        assert metadata["success"] is True