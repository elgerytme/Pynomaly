"""
Comprehensive tests for Import Protocol definitions.
Tests protocol conformance, type checking, and import contracts.
"""

import inspect
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from pynomaly.domain.entities.dataset import Dataset
from pynomaly.shared.protocols.import_protocol import ImportProtocol


class TestImportProtocol:
    """Test suite for ImportProtocol."""

    def test_protocol_is_abstract_base_class(self):
        """Test that ImportProtocol is an abstract base class."""
        assert hasattr(ImportProtocol, "__abstractmethods__")
        assert len(ImportProtocol.__abstractmethods__) > 0

    def test_protocol_defines_required_methods(self):
        """Test that protocol defines all required abstract methods."""
        required_methods = ["import_dataset", "get_supported_formats", "validate_file"]

        for method in required_methods:
            assert hasattr(ImportProtocol, method)
            assert callable(getattr(ImportProtocol, method))

    def test_abstract_methods_are_marked(self):
        """Test that required methods are marked as abstract."""
        abstract_methods = ImportProtocol.__abstractmethods__
        expected_abstract = {"import_dataset", "get_supported_formats", "validate_file"}

        assert abstract_methods == expected_abstract

    def test_protocol_method_signatures(self):
        """Test that protocol methods have correct signatures."""
        # Test import_dataset method signature
        import_sig = inspect.signature(ImportProtocol.import_dataset)
        assert "file_path" in import_sig.parameters
        assert "options" in import_sig.parameters
        assert import_sig.return_annotation == Dataset

        # Test get_supported_formats signature
        formats_sig = inspect.signature(ImportProtocol.get_supported_formats)
        assert formats_sig.return_annotation == list[str]

        # Test validate_file signature
        validate_sig = inspect.signature(ImportProtocol.validate_file)
        assert "file_path" in validate_sig.parameters
        assert validate_sig.return_annotation == bool

    def test_cannot_instantiate_abstract_class(self):
        """Test that ImportProtocol cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ImportProtocol()

    def test_concrete_implementation_example(self):
        """Test a concrete implementation of the import protocol."""

        class ConcreteImporter(ImportProtocol):
            def import_dataset(
                self,
                file_path: str | Path,
                options: dict[str, Any] = None,
            ) -> Dataset:
                # Simplified import implementation
                path = Path(file_path)

                # Mock Dataset creation
                mock_dataset = Mock(spec=Dataset)
                mock_dataset.id = "imported-dataset"
                mock_dataset.name = path.stem
                mock_dataset.metadata = {
                    "source_file": str(path),
                    "format": path.suffix,
                    "import_options": options or {},
                }

                return mock_dataset

            def get_supported_formats(self) -> list[str]:
                return [".csv", ".json", ".xlsx"]

            def validate_file(self, file_path: str | Path) -> bool:
                path = Path(file_path)
                return path.suffix.lower() in [
                    fmt.lower() for fmt in self.get_supported_formats()
                ]

        importer = ConcreteImporter()

        # Test protocol implementation
        assert isinstance(importer, ImportProtocol)

        # Test supported formats
        formats = importer.get_supported_formats()
        assert ".csv" in formats
        assert ".json" in formats
        assert ".xlsx" in formats

        # Test file validation
        assert importer.validate_file("test.csv") is True
        assert importer.validate_file("test.JSON") is True  # Case insensitive
        assert importer.validate_file("test.xml") is False

        # Test import functionality
        dataset = importer.import_dataset("sample.csv")
        assert isinstance(dataset, Mock)
        assert dataset.name == "sample"
        assert dataset.metadata["format"] == ".csv"

    def test_import_with_options(self):
        """Test import protocol with options parameter."""

        class OptionsAwareImporter(ImportProtocol):
            def import_dataset(
                self,
                file_path: str | Path,
                options: dict[str, Any] = None,
            ) -> Dataset:
                path = Path(file_path)

                mock_dataset = Mock(spec=Dataset)
                mock_dataset.id = "options-test"
                mock_dataset.name = path.stem

                metadata = {
                    "source_file": str(path),
                    "format": path.suffix,
                }

                if options:
                    metadata["options_applied"] = True
                    # Simulate applying options
                    if "delimiter" in options:
                        metadata["delimiter"] = options["delimiter"]
                    if "encoding" in options:
                        metadata["encoding"] = options["encoding"]
                    if "skip_rows" in options:
                        metadata["skipped_rows"] = options["skip_rows"]
                    if "column_names" in options:
                        metadata["custom_columns"] = options["column_names"]
                else:
                    metadata["options_applied"] = False

                mock_dataset.metadata = metadata
                return mock_dataset

            def get_supported_formats(self) -> list[str]:
                return [".csv", ".json"]

            def validate_file(self, file_path: str | Path) -> bool:
                return Path(file_path).suffix.lower() in [".csv", ".json"]

        importer = OptionsAwareImporter()

        # Test without options
        dataset_no_options = importer.import_dataset("test.csv")
        assert dataset_no_options.metadata["options_applied"] is False

        # Test with options
        options = {
            "delimiter": ";",
            "encoding": "utf-8",
            "skip_rows": 1,
            "column_names": ["feature_a", "feature_b"],
        }

        dataset_with_options = importer.import_dataset("test.csv", options)
        assert dataset_with_options.metadata["options_applied"] is True
        assert dataset_with_options.metadata["delimiter"] == ";"
        assert dataset_with_options.metadata["encoding"] == "utf-8"
        assert dataset_with_options.metadata["skipped_rows"] == 1
        assert dataset_with_options.metadata["custom_columns"] == [
            "feature_a",
            "feature_b",
        ]

    def test_path_type_handling(self):
        """Test that import protocol handles both string and Path types."""

        class PathHandlingImporter(ImportProtocol):
            def import_dataset(
                self,
                file_path: str | Path,
                options: dict[str, Any] = None,
            ) -> Dataset:
                path = Path(file_path)

                mock_dataset = Mock(spec=Dataset)
                mock_dataset.id = "path-test"
                mock_dataset.name = path.name
                mock_dataset.metadata = {
                    "file_path": str(path),
                    "path_type": type(file_path).__name__,
                    "absolute_path": str(path.absolute()),
                    "file_name": path.name,
                    "file_stem": path.stem,
                    "file_suffix": path.suffix,
                }

                return mock_dataset

            def get_supported_formats(self) -> list[str]:
                return [".csv"]

            def validate_file(self, file_path: str | Path) -> bool:
                return Path(file_path).suffix == ".csv"

        importer = PathHandlingImporter()

        # Test with string path
        dataset_str = importer.import_dataset("test.csv")
        assert dataset_str.metadata["path_type"] == "str"
        assert dataset_str.metadata["file_name"] == "test.csv"
        assert dataset_str.metadata["file_stem"] == "test"
        assert dataset_str.metadata["file_suffix"] == ".csv"

        # Test with Path object
        dataset_path = importer.import_dataset(Path("test.csv"))
        path_type = dataset_path.metadata["path_type"]
        assert path_type in ["PosixPath", "WindowsPath"]  # Platform dependent
        assert dataset_path.metadata["file_name"] == "test.csv"

        # Both should validate correctly
        assert importer.validate_file("test.csv") is True
        assert importer.validate_file(Path("test.csv")) is True

    def test_import_error_handling(self):
        """Test import protocol error handling."""

        class ErrorHandlingImporter(ImportProtocol):
            def import_dataset(
                self,
                file_path: str | Path,
                options: dict[str, Any] = None,
            ) -> Dataset:
                path = Path(file_path)

                # Simulate validation
                if not self.validate_file(file_path):
                    raise ValueError(f"Unsupported file format: {path.suffix}")

                # Simulate import process
                try:
                    # Mock successful import
                    mock_dataset = Mock(spec=Dataset)
                    mock_dataset.id = "error-test"
                    mock_dataset.name = path.stem
                    mock_dataset.metadata = {
                        "source_file": str(path),
                        "import_success": True,
                        "error": None,
                    }

                    return mock_dataset
                except Exception as e:
                    # Return dataset with error information
                    mock_dataset = Mock(spec=Dataset)
                    mock_dataset.id = "error-test"
                    mock_dataset.name = path.stem
                    mock_dataset.metadata = {
                        "source_file": str(path),
                        "import_success": False,
                        "error": str(e),
                    }
                    return mock_dataset

            def get_supported_formats(self) -> list[str]:
                return [".csv", ".json"]

            def validate_file(self, file_path: str | Path) -> bool:
                return Path(file_path).suffix.lower() in [".csv", ".json"]

        importer = ErrorHandlingImporter()

        # Test successful import
        success_dataset = importer.import_dataset("test.csv")
        assert success_dataset.metadata["import_success"] is True
        assert success_dataset.metadata["error"] is None

        # Test validation error
        with pytest.raises(ValueError, match="Unsupported file format"):
            importer.import_dataset("test.xml")

    def test_multiple_format_support(self):
        """Test import protocol with multiple format support."""

        class MultiFormatImporter(ImportProtocol):
            def __init__(self):
                self._format_handlers = {
                    ".csv": self._import_csv,
                    ".json": self._import_json,
                    ".xlsx": self._import_excel,
                }

            def import_dataset(
                self,
                file_path: str | Path,
                options: dict[str, Any] = None,
            ) -> Dataset:
                path = Path(file_path)
                format_ext = path.suffix.lower()

                if format_ext not in self._format_handlers:
                    raise ValueError(f"Unsupported format: {format_ext}")

                handler = self._format_handlers[format_ext]
                return handler(path, options)

            def get_supported_formats(self) -> list[str]:
                return list(self._format_handlers.keys())

            def validate_file(self, file_path: str | Path) -> bool:
                return Path(file_path).suffix.lower() in self._format_handlers

            def _import_csv(
                self, path: Path, options: dict[str, Any] = None
            ) -> Dataset:
                delimiter = options.get("delimiter", ",") if options else ","

                mock_dataset = Mock(spec=Dataset)
                mock_dataset.id = "csv-import"
                mock_dataset.name = path.stem
                mock_dataset.metadata = {
                    "format": "csv",
                    "delimiter": delimiter,
                    "source_file": str(path),
                }

                return mock_dataset

            def _import_json(
                self, path: Path, options: dict[str, Any] = None
            ) -> Dataset:
                mock_dataset = Mock(spec=Dataset)
                mock_dataset.id = "json-import"
                mock_dataset.name = path.stem
                mock_dataset.metadata = {"format": "json", "source_file": str(path)}

                return mock_dataset

            def _import_excel(
                self, path: Path, options: dict[str, Any] = None
            ) -> Dataset:
                sheet_name = (
                    options.get("sheet_name", "Sheet1") if options else "Sheet1"
                )

                mock_dataset = Mock(spec=Dataset)
                mock_dataset.id = "excel-import"
                mock_dataset.name = path.stem
                mock_dataset.metadata = {
                    "format": "excel",
                    "sheet_name": sheet_name,
                    "source_file": str(path),
                }

                return mock_dataset

        importer = MultiFormatImporter()

        # Test CSV import
        csv_dataset = importer.import_dataset("data.csv")
        assert csv_dataset.metadata["format"] == "csv"
        assert csv_dataset.metadata["delimiter"] == ","

        # Test JSON import
        json_dataset = importer.import_dataset("data.json")
        assert json_dataset.metadata["format"] == "json"

        # Test Excel import with options
        excel_dataset = importer.import_dataset(
            "data.xlsx", {"sheet_name": "DataSheet"}
        )
        assert excel_dataset.metadata["format"] == "excel"
        assert excel_dataset.metadata["sheet_name"] == "DataSheet"

        # Test all formats are supported
        supported = importer.get_supported_formats()
        assert ".csv" in supported
        assert ".json" in supported
        assert ".xlsx" in supported

    def test_import_protocol_inheritance(self):
        """Test that import protocol can be properly inherited and extended."""

        class BaseImporter(ImportProtocol):
            def import_dataset(
                self,
                file_path: str | Path,
                options: dict[str, Any] = None,
            ) -> Dataset:
                mock_dataset = Mock(spec=Dataset)
                mock_dataset.id = "base-import"
                mock_dataset.name = Path(file_path).stem
                mock_dataset.metadata = {
                    "base_import": True,
                    "file_path": str(file_path),
                }

                return mock_dataset

            def get_supported_formats(self) -> list[str]:
                return [".txt"]

            def validate_file(self, file_path: str | Path) -> bool:
                return Path(file_path).suffix == ".txt"

        class ExtendedImporter(BaseImporter):
            def import_dataset(
                self,
                file_path: str | Path,
                options: dict[str, Any] = None,
            ) -> Dataset:
                # Call parent implementation
                dataset = super().import_dataset(file_path, options)

                # Add extended functionality
                dataset.metadata.update(
                    {
                        "extended_import": True,
                        "enhanced_features": True,
                    }
                )

                return dataset

            def get_supported_formats(self) -> list[str]:
                # Extend supported formats
                base_formats = super().get_supported_formats()
                return base_formats + [".csv", ".json"]

            def validate_file(self, file_path: str | Path) -> bool:
                return Path(file_path).suffix in self.get_supported_formats()

        extended_importer = ExtendedImporter()

        # Test that it's still an ImportProtocol
        assert isinstance(extended_importer, ImportProtocol)

        # Test extended functionality
        dataset = extended_importer.import_dataset("test.csv")
        assert dataset.metadata["base_import"] is True
        assert dataset.metadata["extended_import"] is True
        assert dataset.metadata["enhanced_features"] is True

        # Test extended format support
        formats = extended_importer.get_supported_formats()
        assert ".txt" in formats
        assert ".csv" in formats
        assert ".json" in formats

        # Test extended validation
        assert extended_importer.validate_file("test.txt") is True
        assert extended_importer.validate_file("test.csv") is True
        assert extended_importer.validate_file("test.xml") is False

    def test_import_protocol_composition(self):
        """Test import protocol can be used in composition patterns."""

        class ImportManager:
            def __init__(self):
                self._importers: dict[str, ImportProtocol] = {}

            def register_importer(
                self, format_ext: str, importer: ImportProtocol
            ) -> None:
                self._importers[format_ext] = importer

            def import_data(
                self,
                file_path: str | Path,
                options: dict[str, Any] = None,
            ) -> Dataset:
                path = Path(file_path)
                format_ext = path.suffix.lower()

                if format_ext not in self._importers:
                    raise ValueError(f"No importer registered for format: {format_ext}")

                importer = self._importers[format_ext]
                return importer.import_dataset(file_path, options)

            def get_supported_formats(self) -> list[str]:
                return list(self._importers.keys())

        # Create specific importers
        class CSVImporter(ImportProtocol):
            def import_dataset(self, file_path, options=None):
                mock_dataset = Mock(spec=Dataset)
                mock_dataset.id = "csv"
                mock_dataset.name = Path(file_path).stem
                mock_dataset.metadata = {"format": "csv", "file_path": str(file_path)}
                return mock_dataset

            def get_supported_formats(self):
                return [".csv"]

            def validate_file(self, file_path):
                return Path(file_path).suffix == ".csv"

        class JSONImporter(ImportProtocol):
            def import_dataset(self, file_path, options=None):
                mock_dataset = Mock(spec=Dataset)
                mock_dataset.id = "json"
                mock_dataset.name = Path(file_path).stem
                mock_dataset.metadata = {"format": "json", "file_path": str(file_path)}
                return mock_dataset

            def get_supported_formats(self):
                return [".json"]

            def validate_file(self, file_path):
                return Path(file_path).suffix == ".json"

        # Test composition
        manager = ImportManager()
        manager.register_importer(".csv", CSVImporter())
        manager.register_importer(".json", JSONImporter())

        # Test CSV import through manager
        csv_dataset = manager.import_data("test.csv")
        assert csv_dataset.metadata["format"] == "csv"

        # Test JSON import through manager
        json_dataset = manager.import_data("test.json")
        assert json_dataset.metadata["format"] == "json"

        # Test supported formats
        supported = manager.get_supported_formats()
        assert ".csv" in supported
        assert ".json" in supported

        # Test unsupported format
        with pytest.raises(ValueError, match="No importer registered for format"):
            manager.import_data("test.xml")

    def test_import_protocol_type_validation(self):
        """Test that import protocol correctly validates types."""

        class TypeValidatingImporter(ImportProtocol):
            def import_dataset(
                self,
                file_path: str | Path,
                options: dict[str, Any] = None,
            ) -> Dataset:
                # Validate file_path type
                if not isinstance(file_path, (str, Path)):
                    raise TypeError(
                        f"file_path must be str or Path, got {type(file_path)}"
                    )

                # Validate options type
                if options is not None and not isinstance(options, dict):
                    raise TypeError(
                        f"options must be dict or None, got {type(options)}"
                    )

                path = Path(file_path)

                mock_dataset = Mock(spec=Dataset)
                mock_dataset.id = "type-validated"
                mock_dataset.name = path.stem
                mock_dataset.metadata = {
                    "file_path_type": type(file_path).__name__,
                    "options_type": type(options).__name__ if options else None,
                    "validation_passed": True,
                }

                return mock_dataset

            def get_supported_formats(self) -> list[str]:
                return [".csv"]

            def validate_file(self, file_path: str | Path) -> bool:
                if not isinstance(file_path, (str, Path)):
                    return False
                return Path(file_path).suffix == ".csv"

        importer = TypeValidatingImporter()

        # Test valid types
        dataset_str = importer.import_dataset("test.csv")
        assert dataset_str.metadata["validation_passed"] is True
        assert dataset_str.metadata["file_path_type"] == "str"

        dataset_path = importer.import_dataset(Path("test.csv"))
        assert dataset_path.metadata["validation_passed"] is True
        path_type = dataset_path.metadata["file_path_type"]
        assert path_type in ["PosixPath", "WindowsPath"]

        dataset_with_options = importer.import_dataset("test.csv", {"key": "value"})
        assert dataset_with_options.metadata["options_type"] == "dict"

        # Test invalid types
        with pytest.raises(TypeError, match="file_path must be str or Path"):
            importer.import_dataset(123)

        with pytest.raises(TypeError, match="options must be dict or None"):
            importer.import_dataset("test.csv", "invalid_options")

        # Test validation with invalid types
        assert importer.validate_file(123) is False
        assert importer.validate_file(None) is False
