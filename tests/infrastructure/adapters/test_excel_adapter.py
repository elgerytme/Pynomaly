"""
Tests for Excel Adapter

Comprehensive test suite for Excel import/export functionality.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from pynomaly.application.dto.export_options import ExportFormat, ExportOptions
from pynomaly.domain.entities.anomaly import Anomaly
from pynomaly.domain.entities.dataset import Dataset
from pynomaly.domain.entities.detection_result import DetectionResult
from pynomaly.domain.value_objects.anomaly_score import AnomalyScore
from pynomaly.infrastructure.adapters.excel_adapter import ExcelAdapter


@pytest.fixture
def sample_detection_result():
    """Create a sample detection result for testing."""
    # Create sample scores and labels
    scores = [
        AnomalyScore(0.1),
        AnomalyScore(0.3),
        AnomalyScore(0.8),
        AnomalyScore(0.9),
        AnomalyScore(0.2),
    ]

    labels = np.array([0, 0, 1, 1, 0])

    # Create anomalies for the anomalous samples
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


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    data = pd.DataFrame(
        {
            "feature_1": [1, 2, 3, 4, 5],
            "feature_2": [10, 20, 30, 40, 50],
            "feature_3": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )

    return Dataset(
        name="test_dataset", data=data, description="Test dataset for Excel adapter"
    )


class TestExcelAdapter:
    """Test cases for Excel adapter functionality."""

    def test_initialization_no_dependencies(self):
        """Test adapter initialization when no Excel libraries are available."""
        with (
            patch(
                "pynomaly.infrastructure.adapters.excel_adapter.OPENPYXL_AVAILABLE",
                False,
            ),
            patch(
                "pynomaly.infrastructure.adapters.excel_adapter.XLSXWRITER_AVAILABLE",
                False,
            ),
        ):
            with pytest.raises(
                ImportError,
                match="Excel adapter requires either openpyxl or xlsxwriter",
            ):
                ExcelAdapter()

    def test_initialization_with_dependencies(self):
        """Test successful adapter initialization with dependencies."""
        with (
            patch(
                "pynomaly.infrastructure.adapters.excel_adapter.OPENPYXL_AVAILABLE",
                True,
            ),
            patch(
                "pynomaly.infrastructure.adapters.excel_adapter.XLSXWRITER_AVAILABLE",
                True,
            ),
        ):
            adapter = ExcelAdapter()
            assert adapter._openpyxl_available is True
            assert adapter._xlsxwriter_available is True

    def test_get_supported_formats(self):
        """Test supported formats method."""
        with (
            patch(
                "pynomaly.infrastructure.adapters.excel_adapter.OPENPYXL_AVAILABLE",
                True,
            ),
            patch(
                "pynomaly.infrastructure.adapters.excel_adapter.XLSXWRITER_AVAILABLE",
                True,
            ),
        ):
            adapter = ExcelAdapter()
            formats = adapter.get_supported_formats()

            assert ".xlsx" in formats
            assert ".xlsm" in formats

    def test_validate_file_success(self):
        """Test file validation for valid Excel files."""
        with (
            patch(
                "pynomaly.infrastructure.adapters.excel_adapter.OPENPYXL_AVAILABLE",
                True,
            ),
            patch(
                "pynomaly.infrastructure.adapters.excel_adapter.XLSXWRITER_AVAILABLE",
                True,
            ),
        ):
            adapter = ExcelAdapter()

            with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp_file:
                tmp_path = Path(tmp_file.name)

                # Mock openpyxl.load_workbook
                with patch("openpyxl.load_workbook") as mock_load:
                    mock_workbook = MagicMock()
                    mock_load.return_value = mock_workbook

                    result = adapter.validate_file(tmp_path)
                    assert result is True
                    mock_workbook.close.assert_called_once()

                # Clean up
                tmp_path.unlink()

    def test_validate_file_invalid_extension(self):
        """Test file validation for invalid file extensions."""
        with (
            patch(
                "pynomaly.infrastructure.adapters.excel_adapter.OPENPYXL_AVAILABLE",
                True,
            ),
            patch(
                "pynomaly.infrastructure.adapters.excel_adapter.XLSXWRITER_AVAILABLE",
                True,
            ),
        ):
            adapter = ExcelAdapter()

            with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp_file:
                tmp_path = Path(tmp_file.name)

                result = adapter.validate_file(tmp_path)
                assert result is False

                # Clean up
                tmp_path.unlink()

    @patch("pynomaly.infrastructure.adapters.excel_adapter.XLSXWRITER_AVAILABLE", True)
    @patch("pynomaly.infrastructure.adapters.excel_adapter.OPENPYXL_AVAILABLE", True)
    def test_export_results_with_xlsxwriter(self, sample_detection_result):
        """Test export results using xlsxwriter."""
        adapter = ExcelAdapter()

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

            # Mock xlsxwriter
            with patch("xlsxwriter.Workbook") as mock_workbook_class:
                mock_workbook = MagicMock()
                mock_worksheet = MagicMock()
                mock_format = MagicMock()

                mock_workbook_class.return_value = mock_workbook
                mock_workbook.add_worksheet.return_value = mock_worksheet
                mock_workbook.add_format.return_value = mock_format
                mock_workbook.add_chart.return_value = MagicMock()

                options = ExportOptions(
                    use_advanced_formatting=True, include_charts=True
                )
                result = adapter.export_results(
                    sample_detection_result, tmp_path, options
                )

                # Verify workbook operations
                mock_workbook_class.assert_called_once_with(str(tmp_path))
                mock_workbook.close.assert_called_once()

                # Verify result metadata
                assert result["file_path"] == str(tmp_path)
                assert result["total_samples"] == 5
                assert result["anomalies_count"] == 2
                assert "export_time" in result
                assert "Results" in result["worksheets"]
                assert "Summary" in result["worksheets"]
                assert "Charts" in result["worksheets"]
                assert "Metadata" in result["worksheets"]

            # Clean up
            tmp_path.unlink()

    @patch("pynomaly.infrastructure.adapters.excel_adapter.XLSXWRITER_AVAILABLE", False)
    @patch("pynomaly.infrastructure.adapters.excel_adapter.OPENPYXL_AVAILABLE", True)
    def test_export_results_with_openpyxl(self, sample_detection_result):
        """Test export results using openpyxl."""
        adapter = ExcelAdapter()

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

            # Mock openpyxl
            with patch("openpyxl.Workbook") as mock_workbook_class:
                mock_workbook = MagicMock()
                mock_worksheet = MagicMock()
                mock_cell = MagicMock()

                mock_workbook_class.return_value = mock_workbook
                mock_workbook.create_sheet.return_value = mock_worksheet
                mock_worksheet.cell.return_value = mock_cell

                options = ExportOptions(use_advanced_formatting=False)
                result = adapter.export_results(
                    sample_detection_result, tmp_path, options
                )

                # Verify workbook operations
                mock_workbook_class.assert_called_once()
                mock_workbook.save.assert_called_once_with(str(tmp_path))

                # Verify result metadata
                assert result["file_path"] == str(tmp_path)
                assert result["total_samples"] == 5
                assert result["anomalies_count"] == 2
                assert "Results" in result["worksheets"]
                assert "Summary" in result["worksheets"]
                assert "Metadata" in result["worksheets"]

            # Clean up
            tmp_path.unlink()

    def test_export_results_no_libraries(self, sample_detection_result):
        """Test export results when no Excel libraries are available."""
        with (
            patch(
                "pynomaly.infrastructure.adapters.excel_adapter.OPENPYXL_AVAILABLE",
                False,
            ),
            patch(
                "pynomaly.infrastructure.adapters.excel_adapter.XLSXWRITER_AVAILABLE",
                False,
            ),
        ):
            adapter = ExcelAdapter()

            with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp_file:
                tmp_path = Path(tmp_file.name)

                with pytest.raises(
                    RuntimeError, match="No Excel library available for export"
                ):
                    adapter.export_results(sample_detection_result, tmp_path)

                # Clean up
                tmp_path.unlink()

    @patch("pynomaly.infrastructure.adapters.excel_adapter.OPENPYXL_AVAILABLE", True)
    @patch("pandas.read_excel")
    def test_import_dataset_success(self, mock_read_excel, sample_dataset):
        """Test successful dataset import from Excel."""
        adapter = ExcelAdapter()

        # Mock pandas.read_excel to return our sample data
        mock_read_excel.return_value = sample_dataset.data

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

            result = adapter.import_dataset(tmp_path)

            # Verify import
            mock_read_excel.assert_called_once()
            assert isinstance(result, Dataset)
            assert result.name == tmp_path.stem
            assert result.n_samples == 5
            assert result.n_features == 3
            assert "source_file" in result.metadata
            assert "import_time" in result.metadata

            # Clean up
            tmp_path.unlink()

    @patch("pynomaly.infrastructure.adapters.excel_adapter.OPENPYXL_AVAILABLE", True)
    @patch("pandas.read_excel")
    def test_import_dataset_with_options(self, mock_read_excel, sample_dataset):
        """Test dataset import with custom options."""
        adapter = ExcelAdapter()

        # Create data with some missing values
        data_with_nan = sample_dataset.data.copy()
        data_with_nan.loc[1, "feature_1"] = np.nan
        mock_read_excel.return_value = data_with_nan

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

            options = {"sheet_name": "Sheet1", "header": 0, "fill_missing": "mean"}

            result = adapter.import_dataset(tmp_path, options)

            # Verify import with options
            mock_read_excel.assert_called_once_with(
                tmp_path, sheet_name="Sheet1", header=0, index_col=None
            )
            assert isinstance(result, Dataset)

            # Clean up
            tmp_path.unlink()

    def test_import_dataset_no_openpyxl(self):
        """Test import dataset when openpyxl is not available."""
        with (
            patch(
                "pynomaly.infrastructure.adapters.excel_adapter.OPENPYXL_AVAILABLE",
                False,
            ),
            patch(
                "pynomaly.infrastructure.adapters.excel_adapter.XLSXWRITER_AVAILABLE",
                True,
            ),
        ):
            adapter = ExcelAdapter()

            with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp_file:
                tmp_path = Path(tmp_file.name)

                with pytest.raises(
                    RuntimeError, match="openpyxl is required for Excel import"
                ):
                    adapter.import_dataset(tmp_path)

                # Clean up
                tmp_path.unlink()

    @patch("pynomaly.infrastructure.adapters.excel_adapter.OPENPYXL_AVAILABLE", True)
    @patch("pandas.read_excel")
    def test_import_dataset_no_numeric_columns(self, mock_read_excel):
        """Test import dataset with no numeric columns."""
        adapter = ExcelAdapter()

        # Create data with only string columns
        text_data = pd.DataFrame(
            {
                "name": ["Alice", "Bob", "Charlie"],
                "city": ["New York", "London", "Tokyo"],
            }
        )
        mock_read_excel.return_value = text_data

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

            with pytest.raises(
                ValueError, match="No numeric columns found for anomaly detection"
            ):
                adapter.import_dataset(tmp_path)

            # Clean up
            tmp_path.unlink()

    @patch("pynomaly.infrastructure.adapters.excel_adapter.OPENPYXL_AVAILABLE", True)
    @patch("pandas.read_excel")
    def test_validate_and_clean_data_drop_missing(self, mock_read_excel):
        """Test data validation and cleaning with drop missing strategy."""
        adapter = ExcelAdapter()

        # Create data with missing values
        data_with_nan = pd.DataFrame(
            {"feature_1": [1, np.nan, 3, 4, 5], "feature_2": [10, 20, np.nan, 40, 50]}
        )

        options = {"fill_missing": "drop"}
        cleaned_data = adapter._validate_and_clean_data(data_with_nan, options)

        # Should drop rows with any NaN values
        assert len(cleaned_data) == 3  # Rows 0, 3, 4 remain
        assert not cleaned_data.isnull().any().any()

    @patch("pynomaly.infrastructure.adapters.excel_adapter.OPENPYXL_AVAILABLE", True)
    def test_validate_and_clean_data_fill_mean(self):
        """Test data validation and cleaning with mean fill strategy."""
        adapter = ExcelAdapter()

        # Create data with missing values
        data_with_nan = pd.DataFrame(
            {"feature_1": [1, np.nan, 3, 4, 5], "feature_2": [10, 20, np.nan, 40, 50]}
        )

        options = {"fill_missing": "mean"}
        cleaned_data = adapter._validate_and_clean_data(data_with_nan, options)

        # Should fill NaN with mean values
        assert len(cleaned_data) == 5
        assert not cleaned_data.isnull().any().any()
        assert cleaned_data.loc[1, "feature_1"] == 3.25  # Mean of [1, 3, 4, 5]
        assert cleaned_data.loc[2, "feature_2"] == 30.0  # Mean of [10, 20, 40, 50]


class TestExportOptions:
    """Test cases for ExportOptions DTO."""

    def test_default_initialization(self):
        """Test default ExportOptions initialization."""
        options = ExportOptions()

        assert options.format == ExportFormat.EXCEL
        assert options.include_charts is True
        assert options.include_summary is True
        assert options.use_advanced_formatting is True
        assert options.sheet_names == ["Results", "Summary", "Charts", "Metadata"]

    def test_for_excel_configuration(self):
        """Test Excel-specific configuration method."""
        options = ExportOptions().for_excel()

        assert options.format == ExportFormat.EXCEL
        assert options.use_advanced_formatting is True
        assert options.highlight_anomalies is True
        assert options.add_conditional_formatting is True
        assert options.include_charts is True

    def test_to_dict_conversion(self):
        """Test conversion to dictionary."""
        options = ExportOptions(
            format=ExportFormat.EXCEL,
            include_charts=False,
            custom_options={"test": "value"},
        )

        result = options.to_dict()

        assert result["format"] == "excel"
        assert result["include_charts"] is False
        assert result["custom_options"] == {"test": "value"}

    def test_from_dict_creation(self):
        """Test creation from dictionary."""
        data = {
            "format": "excel",
            "include_charts": False,
            "use_advanced_formatting": True,
        }

        options = ExportOptions.from_dict(data)

        assert options.format == ExportFormat.EXCEL
        assert options.include_charts is False
        assert options.use_advanced_formatting is True

    def test_invalid_chart_types_filtering(self):
        """Test filtering of invalid chart types."""
        options = ExportOptions(chart_types=["scatter", "invalid_type", "histogram"])

        assert "scatter" in options.chart_types
        assert "histogram" in options.chart_types
        assert "invalid_type" not in options.chart_types

    def test_invalid_permissions_default(self):
        """Test default permissions for invalid values."""
        options = ExportOptions(permissions="invalid")

        assert options.permissions == "view"


@pytest.mark.integration
class TestExcelAdapterIntegration:
    """Integration tests for Excel adapter with real files."""

    @pytest.mark.skipif(
        not hasattr(pytest, "importorskip")
        or pytest.importorskip("openpyxl", reason="openpyxl not available")
        or pytest.importorskip("xlsxwriter", reason="xlsxwriter not available"),
        reason="Excel libraries not available",
    )
    def test_end_to_end_export_import(self, sample_detection_result, sample_dataset):
        """Test complete export-import cycle with real Excel files."""
        adapter = ExcelAdapter()

        with tempfile.TemporaryDirectory() as tmp_dir:
            export_path = Path(tmp_dir) / "test_export.xlsx"
            import_path = Path(tmp_dir) / "test_import.xlsx"

            # Export detection results
            options = ExportOptions(include_charts=True, use_advanced_formatting=True)
            export_result = adapter.export_results(
                sample_detection_result, export_path, options
            )

            assert export_path.exists()
            assert export_result["total_samples"] == 5
            assert export_result["anomalies_count"] == 2

            # Create and save a dataset for import testing
            sample_dataset.data.to_excel(import_path, index=False)

            # Import dataset
            imported_dataset = adapter.import_dataset(import_path)

            assert imported_dataset.n_samples == sample_dataset.n_samples
            assert imported_dataset.n_features == sample_dataset.n_features
            assert list(imported_dataset.feature_names) == list(
                sample_dataset.feature_names
            )
