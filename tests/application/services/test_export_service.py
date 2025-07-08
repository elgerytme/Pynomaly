"""
Tests for Export Service

Comprehensive test suite for the unified export service functionality.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import numpy as np
import pytest
from pynomaly.application.dto.export_options import (
    ExportDestination,
    ExportFormat,
    ExportOptions,
)
from pynomaly.application.services.export_service import ExportService
from pynomaly.domain.entities.anomaly import Anomaly
from pynomaly.domain.entities.detection_result import DetectionResult
from pynomaly.domain.value_objects.anomaly_score import AnomalyScore


@pytest.fixture
def sample_detection_result():
    """Create a sample detection result for testing."""
    scores = [
        AnomalyScore(0.1),
        AnomalyScore(0.3),
        AnomalyScore(0.8),
        AnomalyScore(0.9),
        AnomalyScore(0.2),
    ]

    labels = np.array([0, 0, 1, 1, 0])

    anomalies = [
        Anomaly(
            id=uuid4(),
            index=2,
            score=scores[2],
            feature_values={"feature_1": 10.0, "feature_2": 20.0},
        ),
        Anomaly(
            id=uuid4(),
            index=3,
            score=scores[3],
            feature_values={"feature_1": 15.0, "feature_2": 25.0},
        ),
    ]

    return DetectionResult(
        detector_id=uuid4(),
        dataset_id=uuid4(),
        anomalies=anomalies,
        scores=scores,
        labels=labels,
        threshold=0.5,
        execution_time_ms=150.0,
        metadata={"detector_name": "IsolationForest"},
    )


class TestExportService:
    """Test cases for ExportService functionality."""

    @patch("pynomaly.application.services.export_service.ExcelAdapter")
    @patch("pynomaly.application.services.export_service.PowerBIAdapter")
    @patch("pynomaly.application.services.export_service.GoogleSheetsAdapter")
    @patch("pynomaly.application.services.export_service.SmartsheetAdapter")
    def test_initialization_all_adapters_available(
        self, mock_smartsheet, mock_gsheets, mock_powerbi, mock_excel
    ):
        """Test service initialization when all adapters are available."""

        # Mock successful adapter creation
        mock_excel.return_value = MagicMock()
        mock_powerbi.return_value = MagicMock()
        mock_gsheets.return_value = MagicMock()
        mock_smartsheet.return_value = MagicMock()

        service = ExportService()

        # Verify all adapters were registered
        assert ExportFormat.EXCEL in service._adapters
        assert ExportFormat.POWERBI in service._adapters
        assert ExportFormat.GSHEETS in service._adapters
        assert ExportFormat.SMARTSHEET in service._adapters

        # Verify adapters were instantiated
        mock_excel.assert_called_once()
        mock_powerbi.assert_called_once()
        mock_gsheets.assert_called_once()
        mock_smartsheet.assert_called_once()

    @patch("pynomaly.application.services.export_service.ExcelAdapter")
    @patch("pynomaly.application.services.export_service.PowerBIAdapter")
    @patch("pynomaly.application.services.export_service.GoogleSheetsAdapter")
    @patch("pynomaly.application.services.export_service.SmartsheetAdapter")
    def test_initialization_some_adapters_unavailable(
        self, mock_smartsheet, mock_gsheets, mock_powerbi, mock_excel
    ):
        """Test service initialization when some adapters are unavailable."""

        # Mock successful and failed adapter creation
        mock_excel.return_value = MagicMock()
        mock_powerbi.side_effect = ImportError("PowerBI dependencies not found")
        mock_gsheets.return_value = MagicMock()
        mock_smartsheet.side_effect = ImportError("Smartsheet SDK not found")

        service = ExportService()

        # Verify only available adapters were registered
        assert ExportFormat.EXCEL in service._adapters
        assert ExportFormat.POWERBI not in service._adapters
        assert ExportFormat.GSHEETS in service._adapters
        assert ExportFormat.SMARTSHEET not in service._adapters

        # Verify attempts were made to create all adapters
        mock_excel.assert_called_once()
        mock_powerbi.assert_called_once()
        mock_gsheets.assert_called_once()
        mock_smartsheet.assert_called_once()

    def test_get_supported_formats(self):
        """Test getting supported export formats."""
        with patch(
            "pynomaly.application.services.export_service.ExcelAdapter"
        ) as mock_excel:
            mock_excel.return_value = MagicMock()

            service = ExportService()
            formats = service.get_supported_formats()

            assert ExportFormat.EXCEL in formats
            assert isinstance(formats, list)

    def test_export_results_excel_success(self, sample_detection_result):
        """Test successful export to Excel format."""
        with patch(
            "pynomaly.application.services.export_service.ExcelAdapter"
        ) as mock_adapter_class:
            mock_adapter = MagicMock()
            mock_adapter.validate_file.return_value = True
            mock_adapter.export_results.return_value = {
                "file_path": "/tmp/test.xlsx",
                "total_samples": 5,
                "anomalies_count": 2,
                "export_time": "2024-01-01T12:00:00",
            }
            mock_adapter_class.return_value = mock_adapter

            service = ExportService()

            with tempfile.NamedTemporaryFile(suffix=".xlsx") as tmp_file:
                options = ExportOptions(format=ExportFormat.EXCEL)
                result = service.export_results(
                    sample_detection_result, tmp_file.name, options
                )

                # Verify adapter was called correctly
                mock_adapter.validate_file.assert_called_once()
                mock_adapter.export_results.assert_called_once_with(
                    sample_detection_result, Path(tmp_file.name), options
                )

                # Verify result contains service metadata
                assert result["service"] == "ExportService"
                assert result["format"] == "excel"
                assert result["destination"] == "local_file"
                assert "service_export_time" in result
                assert result["total_samples"] == 5
                assert result["anomalies_count"] == 2

    def test_export_results_unsupported_format(self, sample_detection_result):
        """Test export with unsupported format."""
        with patch("pynomaly.application.services.export_service.ExcelAdapter"):
            service = ExportService()

            # Remove all adapters to simulate unsupported format
            service._adapters.clear()

            options = ExportOptions(format=ExportFormat.EXCEL)

            with pytest.raises(
                ValueError, match="Export format 'excel' is not supported"
            ):
                service.export_results(
                    sample_detection_result, "/tmp/test.xlsx", options
                )

    def test_export_results_validation_failure(self, sample_detection_result):
        """Test export when file validation fails."""
        with patch(
            "pynomaly.application.services.export_service.ExcelAdapter"
        ) as mock_adapter_class:
            mock_adapter = MagicMock()
            mock_adapter.validate_file.return_value = False
            mock_adapter_class.return_value = mock_adapter

            service = ExportService()

            options = ExportOptions(format=ExportFormat.EXCEL)

            with pytest.raises(ValueError, match="Invalid file path for excel export"):
                service.export_results(
                    sample_detection_result, "/invalid/path.xlsx", options
                )

    def test_export_results_adapter_failure(self, sample_detection_result):
        """Test export when adapter throws exception."""
        with patch(
            "pynomaly.application.services.export_service.ExcelAdapter"
        ) as mock_adapter_class:
            mock_adapter = MagicMock()
            mock_adapter.validate_file.return_value = True
            mock_adapter.export_results.side_effect = Exception("Export failed")
            mock_adapter_class.return_value = mock_adapter

            service = ExportService()

            options = ExportOptions(format=ExportFormat.EXCEL)

            with pytest.raises(RuntimeError, match="Export failed: Export failed"):
                service.export_results(
                    sample_detection_result, "/tmp/test.xlsx", options
                )

    def test_validate_export_request_valid(self):
        """Test validation of valid export request."""
        with patch(
            "pynomaly.application.services.export_service.ExcelAdapter"
        ) as mock_adapter_class:
            mock_adapter = MagicMock()
            mock_adapter.validate_file.return_value = True
            mock_adapter.get_supported_formats.return_value = [".xlsx", ".xls"]
            mock_adapter_class.return_value = mock_adapter

            service = ExportService()

            result = service.validate_export_request(
                ExportFormat.EXCEL, "/tmp/test.xlsx"
            )

            assert result["valid"] is True
            assert result["format"] == "excel"
            assert result["file_path"] == "/tmp/test.xlsx"
            assert len(result["errors"]) == 0

    def test_validate_export_request_invalid_format(self):
        """Test validation with unsupported format."""
        with patch("pynomaly.application.services.export_service.ExcelAdapter"):
            service = ExportService()
            service._adapters.clear()  # Remove all adapters

            result = service.validate_export_request(
                ExportFormat.EXCEL, "/tmp/test.xlsx"
            )

            assert result["valid"] is False
            assert len(result["errors"]) > 0
            assert "not supported" in result["errors"][0]

    def test_validate_export_request_invalid_file(self):
        """Test validation with invalid file path."""
        with patch(
            "pynomaly.application.services.export_service.ExcelAdapter"
        ) as mock_adapter_class:
            mock_adapter = MagicMock()
            mock_adapter.validate_file.return_value = False
            mock_adapter_class.return_value = mock_adapter

            service = ExportService()

            result = service.validate_export_request(
                ExportFormat.EXCEL, "/invalid/path.xlsx"
            )

            assert result["valid"] is False
            assert len(result["errors"]) > 0
            assert "Invalid file path" in result["errors"][0]

    def test_create_export_options_excel(self):
        """Test creating Excel-optimized export options."""
        with patch("pynomaly.application.services.export_service.ExcelAdapter"):
            service = ExportService()

            options = service.create_export_options(
                ExportFormat.EXCEL, include_charts=False, highlight_anomalies=True
            )

            assert options.format == ExportFormat.EXCEL
            assert options.use_advanced_formatting is True
            assert options.highlight_anomalies is True
            assert options.add_conditional_formatting is True

    def test_create_export_options_powerbi(self):
        """Test creating Power BI-optimized export options."""
        with patch("pynomaly.application.services.export_service.ExcelAdapter"):
            service = ExportService()

            options = service.create_export_options(
                ExportFormat.POWERBI,
                workspace_id="test-workspace",
                dataset_name="test-dataset",
            )

            assert options.format == ExportFormat.POWERBI
            assert options.workspace_id == "test-workspace"
            assert options.dataset_name == "test-dataset"
            assert options.destination == ExportDestination.API_ENDPOINT

    def test_get_export_statistics(self):
        """Test getting export statistics."""
        with patch(
            "pynomaly.application.services.export_service.ExcelAdapter"
        ) as mock_adapter_class:
            mock_adapter = MagicMock()
            mock_adapter.get_supported_formats.return_value = [".xlsx", ".xls"]
            mock_adapter_class.return_value = mock_adapter

            service = ExportService()

            stats = service.get_export_statistics()

            assert "total_formats" in stats
            assert "supported_formats" in stats
            assert "adapters" in stats
            assert stats["total_formats"] >= 1
            assert "excel" in stats["supported_formats"]
            assert "excel" in stats["adapters"]

    def test_export_multiple_formats(self, sample_detection_result):
        """Test exporting to multiple formats simultaneously."""
        with (
            patch(
                "pynomaly.application.services.export_service.ExcelAdapter"
            ) as mock_excel_class,
            patch(
                "pynomaly.application.services.export_service.PowerBIAdapter"
            ) as mock_powerbi_class,
        ):
            # Mock Excel adapter
            mock_excel = MagicMock()
            mock_excel.validate_file.return_value = True
            mock_excel.get_supported_formats.return_value = [".xlsx"]
            mock_excel.export_results.return_value = {
                "file_path": "/tmp/test.xlsx",
                "success": True,
            }
            mock_excel_class.return_value = mock_excel

            # Mock Power BI adapter
            mock_powerbi = MagicMock()
            mock_powerbi.validate_file.return_value = True
            mock_powerbi.get_supported_formats.return_value = [".powerbi"]
            mock_powerbi.export_results.return_value = {
                "workspace_id": "test-workspace",
                "success": True,
            }
            mock_powerbi_class.return_value = mock_powerbi

            service = ExportService()

            with tempfile.TemporaryDirectory() as tmp_dir:
                base_path = Path(tmp_dir) / "export_test"
                formats = [ExportFormat.EXCEL, ExportFormat.POWERBI]
                options_map = {
                    ExportFormat.POWERBI: ExportOptions().for_powerbi(
                        "test-workspace", "test-dataset"
                    )
                }

                results = service.export_multiple_formats(
                    sample_detection_result, base_path, formats, options_map
                )

                # Verify both formats were exported
                assert ExportFormat.EXCEL in results
                assert ExportFormat.POWERBI in results

                # Verify Excel export
                assert results[ExportFormat.EXCEL]["success"] is True

                # Verify Power BI export
                assert results[ExportFormat.POWERBI]["success"] is True

    def test_export_multiple_formats_with_failure(self, sample_detection_result):
        """Test multiple format export when one format fails."""
        with patch(
            "pynomaly.application.services.export_service.ExcelAdapter"
        ) as mock_excel_class:
            # Mock Excel adapter with failure
            mock_excel = MagicMock()
            mock_excel.validate_file.return_value = True
            mock_excel.get_supported_formats.return_value = [".xlsx"]
            mock_excel.export_results.side_effect = Exception("Export failed")
            mock_excel_class.return_value = mock_excel

            service = ExportService()

            with tempfile.TemporaryDirectory() as tmp_dir:
                base_path = Path(tmp_dir) / "export_test"
                formats = [ExportFormat.EXCEL]

                results = service.export_multiple_formats(
                    sample_detection_result, base_path, formats
                )

                # Verify failure was captured
                assert ExportFormat.EXCEL in results
                assert results[ExportFormat.EXCEL]["success"] is False
                assert "error" in results[ExportFormat.EXCEL]

    def test_get_supported_file_extensions(self):
        """Test getting supported file extensions for specific format."""
        with patch(
            "pynomaly.application.services.export_service.ExcelAdapter"
        ) as mock_adapter_class:
            mock_adapter = MagicMock()
            mock_adapter.get_supported_formats.return_value = [".xlsx", ".xls"]
            mock_adapter_class.return_value = mock_adapter

            service = ExportService()

            extensions = service.get_supported_file_extensions(ExportFormat.EXCEL)

            assert ".xlsx" in extensions
            assert ".xls" in extensions

    def test_get_supported_file_extensions_unsupported_format(self):
        """Test getting file extensions for unsupported format."""
        with patch("pynomaly.application.services.export_service.ExcelAdapter"):
            service = ExportService()
            service._adapters.clear()  # Remove all adapters

            with pytest.raises(
                ValueError, match="Export format 'excel' is not supported"
            ):
                service.get_supported_file_extensions(ExportFormat.EXCEL)


class TestExportServiceIntegration:
    """Integration tests for ExportService with real adapters."""

    @pytest.mark.integration
    def test_excel_export_integration(self, sample_detection_result):
        """Test integration with real Excel adapter."""
        # This test would require actual Excel libraries
        # Skip if not available
        pytest.importorskip("openpyxl", reason="openpyxl not available")

        service = ExportService()

        # Check if Excel adapter is available
        if ExportFormat.EXCEL not in service.get_supported_formats():
            pytest.skip("Excel adapter not available")

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp_file:
            options = ExportOptions(format=ExportFormat.EXCEL)

            try:
                result = service.export_results(
                    sample_detection_result, tmp_file.name, options
                )

                # Verify export succeeded
                assert result["success"] is True
                assert result["total_samples"] == 5
                assert result["anomalies_count"] == 2
                assert Path(tmp_file.name).exists()

            finally:
                # Clean up
                if Path(tmp_file.name).exists():
                    Path(tmp_file.name).unlink()

    @pytest.mark.integration
    def test_service_statistics_integration(self):
        """Test getting real service statistics."""
        service = ExportService()
        stats = service.get_export_statistics()

        # Verify statistics structure
        assert "total_formats" in stats
        assert "supported_formats" in stats
        assert "adapters" in stats
        assert isinstance(stats["total_formats"], int)
        assert isinstance(stats["supported_formats"], list)
        assert isinstance(stats["adapters"], dict)

        # At least Excel should be available in most environments
        if stats["total_formats"] > 0:
            assert len(stats["supported_formats"]) > 0
            assert len(stats["adapters"]) > 0
