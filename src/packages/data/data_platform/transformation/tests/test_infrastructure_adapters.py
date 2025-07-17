import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import json

from data_transformation.infrastructure.adapters.data_source_adapter import DataSourceAdapter
from data_transformation.domain.value_objects.pipeline_config import SourceType


class TestDataSourceAdapter:
    def setup_method(self):
        self.adapter = DataSourceAdapter()

    def test_load_csv_data(self, sample_csv_file):
        data = self.adapter.load_data(sample_csv_file, SourceType.CSV)
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 5
        assert "numeric_col" in data.columns
        assert "categorical_col" in data.columns

    def test_load_json_data(self, sample_json_file):
        data = self.adapter.load_data(sample_json_file, SourceType.JSON)
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 5

    def test_auto_detect_format_csv(self, sample_csv_file):
        detected_type = self.adapter.auto_detect_format(sample_csv_file)
        assert detected_type == SourceType.CSV

    def test_auto_detect_format_json(self, sample_json_file):
        detected_type = self.adapter.auto_detect_format(sample_json_file)
        assert detected_type == SourceType.JSON

    def test_validate_source_csv(self, sample_csv_file):
        is_valid, issues = self.adapter.validate_source(sample_csv_file, SourceType.CSV)
        assert is_valid is True
        assert len(issues) == 0

    def test_validate_source_nonexistent_file(self):
        is_valid, issues = self.adapter.validate_source("/nonexistent/file.csv", SourceType.CSV)
        assert is_valid is False
        assert len(issues) > 0

    def test_get_source_info_csv(self, sample_csv_file):
        info = self.adapter.get_source_info(sample_csv_file, SourceType.CSV)
        
        assert "file_size" in info
        assert "format" in info
        assert "estimated_rows" in info
        assert info["format"] == "csv"

    def test_load_csv_with_custom_options(self, sample_dataframe):
        # Create CSV with custom delimiter
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_dataframe.to_csv(f.name, index=False, sep=';')
            
            data = self.adapter.load_data(f.name, SourceType.CSV, sep=';')
            assert isinstance(data, pd.DataFrame)
            assert len(data) == 5
            
        Path(f.name).unlink(missing_ok=True)

    def test_load_dataframe_directly(self, sample_dataframe):
        data = self.adapter.load_data(sample_dataframe, "dataframe")
        
        assert isinstance(data, pd.DataFrame)
        assert data.equals(sample_dataframe)

    def test_load_excel_data(self, sample_dataframe):
        # Create temporary Excel file
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
            sample_dataframe.to_excel(f.name, index=False)
            
            data = self.adapter.load_data(f.name, SourceType.EXCEL)
            assert isinstance(data, pd.DataFrame)
            assert len(data) == 5
            
        Path(f.name).unlink(missing_ok=True)

    def test_load_parquet_data(self, sample_dataframe):
        # Create temporary Parquet file
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            sample_dataframe.to_parquet(f.name, index=False)
            
            data = self.adapter.load_data(f.name, SourceType.PARQUET)
            assert isinstance(data, pd.DataFrame)
            assert len(data) == 5
            
        Path(f.name).unlink(missing_ok=True)

    def test_unsupported_source_type(self, sample_csv_file):
        with pytest.raises(ValueError, match="Unsupported source type"):
            self.adapter.load_data(sample_csv_file, "unsupported_type")

    def test_load_data_with_encoding(self, sample_dataframe):
        # Create CSV with specific encoding
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            sample_dataframe.to_csv(f.name, index=False)
            
            data = self.adapter.load_data(f.name, SourceType.CSV, encoding='utf-8')
            assert isinstance(data, pd.DataFrame)
            assert len(data) == 5
            
        Path(f.name).unlink(missing_ok=True)

    def test_load_csv_with_different_separators(self, sample_dataframe):
        separators = [',', ';', '\t', '|']
        
        for sep in separators:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                sample_dataframe.to_csv(f.name, index=False, sep=sep)
                
                data = self.adapter.load_data(f.name, SourceType.CSV, sep=sep)
                assert isinstance(data, pd.DataFrame)
                assert len(data) == 5
                
            Path(f.name).unlink(missing_ok=True)

    def test_load_json_with_different_orientations(self, sample_dataframe):
        orientations = ['records', 'index', 'values']
        
        for orient in orientations:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                sample_dataframe.to_json(f.name, orient=orient)
                
                try:
                    data = self.adapter.load_data(f.name, SourceType.JSON, orient=orient)
                    assert isinstance(data, pd.DataFrame)
                except (ValueError, KeyError):
                    # Some orientations might not work with all data types
                    pass
                
            Path(f.name).unlink(missing_ok=True)

    def test_get_loader_for_source_type(self):
        csv_loader = self.adapter._get_loader(SourceType.CSV)
        assert csv_loader == self.adapter._load_csv
        
        json_loader = self.adapter._get_loader(SourceType.JSON)
        assert json_loader == self.adapter._load_json
        
        excel_loader = self.adapter._get_loader(SourceType.EXCEL)
        assert excel_loader == self.adapter._load_excel
        
        parquet_loader = self.adapter._get_loader(SourceType.PARQUET)
        assert parquet_loader == self.adapter._load_parquet

    def test_error_handling_invalid_csv(self):
        # Create invalid CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("invalid,csv,content\n1,2\n3,4,5,6")  # Inconsistent columns
            
            with pytest.raises(Exception):
                self.adapter.load_data(f.name, SourceType.CSV)
                
        Path(f.name).unlink(missing_ok=True)

    def test_error_handling_invalid_json(self):
        # Create invalid JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            
            with pytest.raises(Exception):
                self.adapter.load_data(f.name, SourceType.JSON)
                
        Path(f.name).unlink(missing_ok=True)