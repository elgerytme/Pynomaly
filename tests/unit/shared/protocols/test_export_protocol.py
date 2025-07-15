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
        """Test protocol has required abstract methods."""
        assert hasattr(ExportProtocol, "export_results")
        assert hasattr(ExportProtocol, "get_supported_formats")
        assert hasattr(ExportProtocol, "validate_file")

    def test_abstract_base_class(self):
        """Test ExportProtocol is an abstract base class."""
        # Should not be able to instantiate abstract class
        with pytest.raises(TypeError):
            ExportProtocol()

    def test_export_results_method_signature(self):
        """Test export_results method has correct signature."""

        class ConcreteExporter(ExportProtocol):
            def export_results(
                self,
                results: DetectionResult,
                file_path: str | Path,
                options: ExportOptions = None,
            ) -> dict[str, Any]:
                return {"status": "success", "exported_rows": 100}

            def get_supported_formats(self) -> list[str]:
                return [".csv"]

            def validate_file(self, file_path: str | Path) -> bool:
                return True

        exporter = ConcreteExporter()
        mock_results = Mock(spec=DetectionResult)
        mock_options = Mock(spec=ExportOptions)

        # Test with string file path
        result = exporter.export_results(mock_results, "output.csv", mock_options)
        assert isinstance(result, dict)

        # Test with Path object
        result = exporter.export_results(mock_results, Path("output.csv"), mock_options)
        assert isinstance(result, dict)

        # Test with None options
        result = exporter.export_results(mock_results, "output.csv", None)
        assert isinstance(result, dict)

    def test_get_supported_formats_method_signature(self):
        """Test get_supported_formats method has correct signature."""

        class ConcreteExporter(ExportProtocol):
            def export_results(
                self,
                results: DetectionResult,
                file_path: str | Path,
                options: ExportOptions = None,
            ) -> dict[str, Any]:
                return {}

            def get_supported_formats(self) -> list[str]:
                return [".csv", ".xlsx", ".json"]

            def validate_file(self, file_path: str | Path) -> bool:
                return True

        exporter = ConcreteExporter()
        formats = exporter.get_supported_formats()

        assert isinstance(formats, list)
        assert all(isinstance(fmt, str) for fmt in formats)
        assert all(fmt.startswith(".") for fmt in formats)

    def test_validate_file_method_signature(self):
        """Test validate_file method has correct signature."""

        class ConcreteExporter(ExportProtocol):
            def export_results(
                self,
                results: DetectionResult,
                file_path: str | Path,
                options: ExportOptions = None,
            ) -> dict[str, Any]:
                return {}

            def get_supported_formats(self) -> list[str]:
                return [".csv"]

            def validate_file(self, file_path: str | Path) -> bool:
                if isinstance(file_path, (str, Path)):
                    return str(file_path).endswith(".csv")
                return False

        exporter = ConcreteExporter()

        # Test with string path
        assert exporter.validate_file("data.csv") is True
        assert exporter.validate_file("data.txt") is False

        # Test with Path object
        assert exporter.validate_file(Path("data.csv")) is True
        assert exporter.validate_file(Path("data.txt")) is False

    def test_concrete_implementation(self):
        """Test concrete implementation works correctly."""

        class CSVExporter(ExportProtocol):
            def export_results(
                self,
                results: DetectionResult,
                file_path: str | Path,
                options: ExportOptions = None,
            ) -> dict[str, Any]:
                # Simulate exporting CSV
                return {
                    "format": "csv",
                    "file_path": str(file_path),
                    "exported_rows": 100,
                    "exported_columns": 5,
                    "status": "success",
                }

            def get_supported_formats(self) -> list[str]:
                return [".csv"]

            def validate_file(self, file_path: str | Path) -> bool:
                return str(file_path).lower().endswith(".csv")

        exporter = CSVExporter()
        mock_results = Mock(spec=DetectionResult)

        # Test export
        result = exporter.export_results(mock_results, "output.csv")
        assert result["format"] == "csv"
        assert result["status"] == "success"
        assert result["exported_rows"] == 100

        # Test format support
        formats = exporter.get_supported_formats()
        assert ".csv" in formats

        # Test validation
        assert exporter.validate_file("data.csv") is True
        assert exporter.validate_file("data.CSV") is True
        assert exporter.validate_file("data.xlsx") is False

    def test_multiple_format_exporter(self):
        """Test exporter supporting multiple formats."""

        class MultiFormatExporter(ExportProtocol):
            def export_results(
                self,
                results: DetectionResult,
                file_path: str | Path,
                options: ExportOptions = None,
            ) -> dict[str, Any]:
                file_str = str(file_path)
                if file_str.lower().endswith(".csv"):
                    format_type = "csv"
                elif file_str.lower().endswith(".xlsx"):
                    format_type = "excel"
                elif file_str.lower().endswith(".json"):
                    format_type = "json"
                else:
                    format_type = "unknown"

                return {
                    "format": format_type,
                    "file_path": file_str,
                    "status": "success" if format_type != "unknown" else "error",
                }

            def get_supported_formats(self) -> list[str]:
                return [".csv", ".xlsx", ".json"]

            def validate_file(self, file_path: str | Path) -> bool:
                file_str = str(file_path).lower()
                return any(
                    file_str.endswith(fmt.lower())
                    for fmt in self.get_supported_formats()
                )

        exporter = MultiFormatExporter()
        mock_results = Mock(spec=DetectionResult)

        # Test different formats
        csv_result = exporter.export_results(mock_results, "output.csv")
        assert csv_result["format"] == "csv"

        excel_result = exporter.export_results(mock_results, "output.xlsx")
        assert excel_result["format"] == "excel"

        json_result = exporter.export_results(mock_results, "output.json")
        assert json_result["format"] == "json"

        # Test validation
        assert exporter.validate_file("data.csv") is True
        assert exporter.validate_file("data.xlsx") is True
        assert exporter.validate_file("data.json") is True
        assert exporter.validate_file("data.txt") is False

    def test_exporter_with_options(self):
        """Test exporter that uses export options."""

        class ConfigurableExporter(ExportProtocol):
            def export_results(
                self,
                results: DetectionResult,
                file_path: str | Path,
                options: ExportOptions = None,
            ) -> dict[str, Any]:
                result = {
                    "file_path": str(file_path),
                    "status": "success",
                }

                if options:
                    # Mock accessing options (would normally use actual options attributes)
                    result["used_options"] = True
                    result["options_type"] = type(options).__name__
                else:
                    result["used_options"] = False

                return result

            def get_supported_formats(self) -> list[str]:
                return [".csv"]

            def validate_file(self, file_path: str | Path) -> bool:
                return True

        exporter = ConfigurableExporter()
        mock_results = Mock(spec=DetectionResult)
        mock_options = Mock(spec=ExportOptions)

        # Test with options
        result_with_options = exporter.export_results(
            mock_results, "output.csv", mock_options
        )
        assert result_with_options["used_options"] is True

        # Test without options
        result_without_options = exporter.export_results(
            mock_results, "output.csv", None
        )
        assert result_without_options["used_options"] is False

    def test_exporter_error_handling(self):
        """Test exporter error handling."""

        class ErrorHandlingExporter(ExportProtocol):
            def export_results(
                self,
                results: DetectionResult,
                file_path: str | Path,
                options: ExportOptions = None,
            ) -> dict[str, Any]:
                if str(file_path) == "invalid.txt":
                    raise ValueError("Unsupported file format")

                return {
                    "file_path": str(file_path),
                    "status": "success",
                }

            def get_supported_formats(self) -> list[str]:
                return [".csv"]

            def validate_file(self, file_path: str | Path) -> bool:
                return str(file_path).endswith(".csv")

        exporter = ErrorHandlingExporter()
        mock_results = Mock(spec=DetectionResult)

        # Test successful export
        result = exporter.export_results(mock_results, "output.csv")
        assert result["status"] == "success"

        # Test error case
        with pytest.raises(ValueError, match="Unsupported file format"):
            exporter.export_results(mock_results, "invalid.txt")

    def test_inheritance_behavior(self):
        """Test that subclasses must implement all abstract methods."""
        # Test incomplete implementation
        with pytest.raises(TypeError):

            class IncompleteExporter(ExportProtocol):
                def export_results(
                    self,
                    results: DetectionResult,
                    file_path: str | Path,
                    options: ExportOptions = None,
                ) -> dict[str, Any]:
                    return {}

                # Missing get_supported_formats and validate_file

            IncompleteExporter()

    def test_method_contracts(self):
        """Test method contracts and return types."""

        class ContractTestExporter(ExportProtocol):
            def export_results(
                self,
                results: DetectionResult,
                file_path: str | Path,
                options: ExportOptions = None,
            ) -> dict[str, Any]:
                # Must return dict[str, Any]
                return {"key": "value", "number": 42, "boolean": True}

            def get_supported_formats(self) -> list[str]:
                # Must return list[str]
                return [".csv", ".xlsx"]

            def validate_file(self, file_path: str | Path) -> bool:
                # Must return bool
                return True

        exporter = ContractTestExporter()
        mock_results = Mock(spec=DetectionResult)

        # Test return types
        export_result = exporter.export_results(mock_results, "test.csv")
        assert isinstance(export_result, dict)
        assert all(isinstance(key, str) for key in export_result.keys())

        formats = exporter.get_supported_formats()
        assert isinstance(formats, list)
        assert all(isinstance(fmt, str) for fmt in formats)

        is_valid = exporter.validate_file("test.csv")
        assert isinstance(is_valid, bool)
