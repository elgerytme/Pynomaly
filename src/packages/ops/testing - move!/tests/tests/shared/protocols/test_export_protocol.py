"""
Tests for the export protocol implementation.

This module tests the ExportProtocol to ensure proper contract enforcement
and runtime behavior checking for all export implementations.
"""

from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pytest

from pynomaly.application.dto.export_options import ExportOptions
from pynomaly.domain.entities import Anomaly, DetectionResult
from pynomaly.domain.value_objects import AnomalyScore
from pynomaly.shared.protocols.export_protocol import ExportProtocol


class MockExportOptions:
    """Mock export options for testing."""

    def __init__(self, format: str = "csv", include_metadata: bool = True):
        self.format = format
        self.include_metadata = include_metadata
        self.encoding = "utf-8"
        self.delimiter = ","


class MockExporter(ExportProtocol):
    """Mock implementation of ExportProtocol for testing."""

    def __init__(self):
        self.supported_formats = [".csv", ".xlsx", ".json", ".parquet"]
        self.export_count = 0
        self.last_export_path = None
        self.last_export_results = None
        self.should_fail_validation = False
        self.should_fail_export = False

    def export_results(
        self,
        results: DetectionResult,
        file_path: str | Path,
        options: ExportOptions = None,
    ) -> dict[str, Any]:
        """Mock export implementation."""
        if self.should_fail_export:
            raise ValueError("Export failed as requested")

        self.export_count += 1
        self.last_export_path = str(file_path)
        self.last_export_results = results

        # Simulate export metadata
        return {
            "status": "success",
            "file_path": str(file_path),
            "format": Path(file_path).suffix,
            "total_records": len(results.anomalies) if results.anomalies else 0,
            "export_timestamp": "2025-07-11T10:30:00Z",
            "file_size_bytes": 1024,
            "encoding": options.encoding
            if options and hasattr(options, "encoding")
            else "utf-8",
        }

    def get_supported_formats(self) -> list[str]:
        """Return supported formats."""
        return self.supported_formats

    def validate_file(self, file_path: str | Path) -> bool:
        """Validate file path."""
        if self.should_fail_validation:
            return False

        path = Path(file_path)
        return path.suffix in self.supported_formats


class TestExportProtocol:
    """Test suite for ExportProtocol contract enforcement."""

    @pytest.fixture
    def sample_detection_result(self):
        """Create sample detection result for testing."""
        anomalies = [
            Anomaly(
                score=AnomalyScore(value=0.8 + i * 0.05),
                data_point={"feature_1": i, "feature_2": i * 2, "index": i},
                detector_name="test_detector",
            )
            for i in range(5)
        ]

        return DetectionResult(
            detector_id=uuid4(),
            dataset_id=uuid4(),
            anomalies=anomalies,
            scores=[
                AnomalyScore(value=0.1),
                AnomalyScore(value=0.9),
                AnomalyScore(value=0.3),
            ],
            labels=np.array([False, True, False]),
            threshold=0.5,
            metadata={
                "algorithm": "isolation_forest",
                "execution_time_ms": 150.0,
                "dataset_size": 1000,
            },
        )

    @pytest.fixture
    def mock_exporter(self):
        """Create mock exporter for testing."""
        return MockExporter()

    @pytest.fixture
    def export_options(self):
        """Create mock export options."""
        return MockExportOptions()

    def test_protocol_compliance(self, mock_exporter):
        """Test that mock exporter implements the protocol correctly."""
        assert isinstance(mock_exporter, ExportProtocol)

        # Check that all required methods exist
        assert hasattr(mock_exporter, "export_results")
        assert hasattr(mock_exporter, "get_supported_formats")
        assert hasattr(mock_exporter, "validate_file")

        # Check method signatures
        assert callable(mock_exporter.export_results)
        assert callable(mock_exporter.get_supported_formats)
        assert callable(mock_exporter.validate_file)

    def test_export_results_basic_functionality(
        self, mock_exporter, sample_detection_result, export_options
    ):
        """Test basic export results functionality."""
        file_path = "/tmp/test_export.csv"

        # Test successful export
        result = mock_exporter.export_results(
            sample_detection_result, file_path, export_options
        )

        # Verify return structure
        assert isinstance(result, dict)
        assert "status" in result
        assert result["status"] == "success"
        assert "file_path" in result
        assert result["file_path"] == file_path
        assert "total_records" in result
        assert result["total_records"] == 5  # Number of anomalies

        # Verify exporter state
        assert mock_exporter.export_count == 1
        assert mock_exporter.last_export_path == file_path
        assert mock_exporter.last_export_results == sample_detection_result

    def test_export_results_with_different_formats(
        self, mock_exporter, sample_detection_result
    ):
        """Test export with different file formats."""
        formats = [".csv", ".xlsx", ".json", ".parquet"]

        for format_ext in formats:
            file_path = f"/tmp/test_export{format_ext}"
            result = mock_exporter.export_results(sample_detection_result, file_path)

            assert result["format"] == format_ext
            assert result["file_path"] == file_path

    def test_export_results_without_options(
        self, mock_exporter, sample_detection_result
    ):
        """Test export without providing options."""
        file_path = "/tmp/test_export_no_options.csv"

        # Should work with None options
        result = mock_exporter.export_results(sample_detection_result, file_path, None)

        assert result["status"] == "success"
        assert result["file_path"] == file_path

    def test_export_results_with_path_object(
        self, mock_exporter, sample_detection_result
    ):
        """Test export with Path object instead of string."""
        file_path = Path("/tmp/test_export_path.csv")

        result = mock_exporter.export_results(sample_detection_result, file_path)

        assert result["status"] == "success"
        assert result["file_path"] == str(file_path)

    def test_export_results_error_handling(
        self, mock_exporter, sample_detection_result
    ):
        """Test export error handling."""
        mock_exporter.should_fail_export = True

        with pytest.raises(ValueError, match="Export failed as requested"):
            mock_exporter.export_results(sample_detection_result, "/tmp/fail.csv")

    def test_get_supported_formats(self, mock_exporter):
        """Test supported formats retrieval."""
        formats = mock_exporter.get_supported_formats()

        assert isinstance(formats, list)
        assert len(formats) > 0
        assert all(isinstance(fmt, str) for fmt in formats)
        assert all(fmt.startswith(".") for fmt in formats)

        # Check expected formats
        expected_formats = [".csv", ".xlsx", ".json", ".parquet"]
        for fmt in expected_formats:
            assert fmt in formats

    def test_validate_file_success(self, mock_exporter):
        """Test successful file validation."""
        valid_paths = [
            "/tmp/test.csv",
            Path("/tmp/test.xlsx"),
            "/data/export.json",
            Path("/output/results.parquet"),
        ]

        for path in valid_paths:
            assert mock_exporter.validate_file(path) is True

    def test_validate_file_failure(self, mock_exporter):
        """Test file validation failure."""
        invalid_paths = [
            "/tmp/test.txt",  # Unsupported format
            "/tmp/test.pdf",  # Unsupported format
            "/tmp/test",  # No extension
            Path("/tmp/test.doc"),  # Unsupported format
        ]

        for path in invalid_paths:
            assert mock_exporter.validate_file(path) is False

    def test_validate_file_forced_failure(self, mock_exporter):
        """Test file validation when forced to fail."""
        mock_exporter.should_fail_validation = True

        # Even valid formats should fail
        assert mock_exporter.validate_file("/tmp/test.csv") is False
        assert mock_exporter.validate_file(Path("/tmp/test.xlsx")) is False

    def test_export_empty_results(self, mock_exporter):
        """Test export with empty detection results."""
        empty_result = DetectionResult(
            detector_id=uuid4(),
            dataset_id=uuid4(),
            anomalies=[],
            scores=[],
            labels=np.array([]),
            threshold=0.5,
            metadata={"algorithm": "test"},
        )

        result = mock_exporter.export_results(empty_result, "/tmp/empty.csv")

        assert result["status"] == "success"
        assert result["total_records"] == 0

    def test_export_large_results(self, mock_exporter):
        """Test export with large number of anomalies."""
        # Create many anomalies
        anomalies = [
            Anomaly(
                score=AnomalyScore(value=0.5 + (i % 50) * 0.01),
                data_point={"value": i, "index": i},
                detector_name="test_detector",
            )
            for i in range(1000)
        ]

        large_result = DetectionResult(
            detector_id=uuid4(),
            dataset_id=uuid4(),
            anomalies=anomalies,
            scores=[AnomalyScore(value=0.5)] * 1000,
            labels=np.array([True] * 1000),
            threshold=0.5,
            metadata={"algorithm": "test"},
        )

        result = mock_exporter.export_results(large_result, "/tmp/large.csv")

        assert result["status"] == "success"
        assert result["total_records"] == 1000

    def test_multiple_exports(self, mock_exporter, sample_detection_result):
        """Test multiple consecutive exports."""
        file_paths = ["/tmp/export1.csv", "/tmp/export2.xlsx", "/tmp/export3.json"]

        for i, path in enumerate(file_paths, 1):
            result = mock_exporter.export_results(sample_detection_result, path)

            assert result["status"] == "success"
            assert mock_exporter.export_count == i
            assert mock_exporter.last_export_path == path

    def test_export_with_complex_options(self, mock_exporter, sample_detection_result):
        """Test export with complex options."""
        complex_options = MockExportOptions(format="xlsx", include_metadata=True)
        complex_options.encoding = "utf-16"
        complex_options.compression = "gzip"
        complex_options.include_headers = True

        result = mock_exporter.export_results(
            sample_detection_result, "/tmp/complex.xlsx", complex_options
        )

        assert result["status"] == "success"
        assert result["encoding"] == "utf-16"


class IncompleteExporter:
    """Incomplete implementation missing required methods."""

    def export_results(self, results, file_path, options=None):
        return {"status": "success"}

    # Missing get_supported_formats and validate_file


class TestExportProtocolEnforcement:
    """Test protocol enforcement and runtime checking."""

    def test_incomplete_implementation_detection(self):
        """Test that incomplete implementations are detected."""
        incomplete = IncompleteExporter()

        # Should not be considered a valid implementation
        assert not isinstance(incomplete, ExportProtocol)

    def test_protocol_method_signatures(self):
        """Test that protocol defines correct method signatures."""
        # This test ensures the protocol is properly defined
        methods = ExportProtocol.__abstractmethods__

        expected_methods = {"export_results", "get_supported_formats", "validate_file"}
        assert methods == expected_methods

    def test_protocol_runtime_checking(self):
        """Test runtime protocol checking works correctly."""
        mock_exporter = MockExporter()

        # Should pass runtime check
        assert isinstance(mock_exporter, ExportProtocol)

        # Protocol methods should be callable
        assert callable(mock_exporter.export_results)
        assert callable(mock_exporter.get_supported_formats)
        assert callable(mock_exporter.validate_file)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
