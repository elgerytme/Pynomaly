"""Data source adapter for loading data from various sources."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
import json

import pandas as pd
import numpy as np

from ...domain.value_objects.pipeline_config import SourceType

logger = logging.getLogger(__name__)


class DataSourceAdapter:
    """
    Adapter for loading data from various data sources.
    
    Supports multiple data formats including CSV, JSON, Parquet, Excel,
    and provides extensible interface for custom data loaders.
    """
    
    def __init__(self) -> None:
        """Initialize the data source adapter."""
        self._loaders = {
            SourceType.CSV: self._load_csv,
            SourceType.JSON: self._load_json,
            SourceType.PARQUET: self._load_parquet,
            SourceType.EXCEL: self._load_excel,
        }
    
    def load_data(
        self,
        source_path: str,
        source_type: Union[str, SourceType],
        **kwargs: Any
    ) -> pd.DataFrame:
        """
        Load data from the specified source.
        
        Args:
            source_path: Path to the data source
            source_type: Type of data source
            **kwargs: Additional parameters for the specific loader
            
        Returns:
            Loaded data as pandas DataFrame
            
        Raises:
            ValueError: If source type is not supported
            FileNotFoundError: If source file doesn't exist
        """
        # Convert string to SourceType if needed
        if isinstance(source_type, str):
            try:
                source_type = SourceType(source_type.lower())
            except ValueError:
                raise ValueError(f"Unsupported source type: {source_type}")
        
        logger.info(f"Loading data from {source_path} (type: {source_type})")
        
        # Check if file exists
        if not Path(source_path).exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")
        
        # Get appropriate loader
        loader = self._loaders.get(source_type)
        if loader is None:
            raise ValueError(f"No loader available for source type: {source_type}")
        
        try:
            # Load data using appropriate loader
            data = loader(source_path, **kwargs)
            
            logger.info(f"Successfully loaded data: {data.shape[0]} rows, {data.shape[1]} columns")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load data from {source_path}: {str(e)}")
            raise
    
    def _load_csv(self, file_path: str, **kwargs: Any) -> pd.DataFrame:
        """Load data from CSV file."""
        # Set default parameters
        default_params = {
            "encoding": "utf-8",
            "low_memory": False,
            "na_values": ["", "NA", "N/A", "null", "NULL", "none", "None"],
        }
        
        # Merge with user parameters
        params = {**default_params, **kwargs}
        
        try:
            # Try to detect separator if not provided
            if "sep" not in params and "delimiter" not in params:
                params["sep"] = self._detect_csv_separator(file_path)
            
            data = pd.read_csv(file_path, **params)
            
            # Basic data cleaning
            data = self._basic_csv_cleaning(data)
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading CSV file {file_path}: {str(e)}")
            raise
    
    def _load_json(self, file_path: str, **kwargs: Any) -> pd.DataFrame:
        """Load data from JSON file."""
        default_params = {
            "orient": "records",
            "lines": False,
        }
        
        params = {**default_params, **kwargs}
        
        try:
            # Try to load as JSON Lines first, then regular JSON
            if params.get("lines", False):
                data = pd.read_json(file_path, lines=True, **{k: v for k, v in params.items() if k != "lines"})
            else:
                data = pd.read_json(file_path, **params)
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading JSON file {file_path}: {str(e)}")
            raise
    
    def _load_parquet(self, file_path: str, **kwargs: Any) -> pd.DataFrame:
        """Load data from Parquet file."""
        try:
            data = pd.read_parquet(file_path, **kwargs)
            return data
            
        except Exception as e:
            logger.error(f"Error loading Parquet file {file_path}: {str(e)}")
            raise
    
    def _load_excel(self, file_path: str, **kwargs: Any) -> pd.DataFrame:
        """Load data from Excel file."""
        default_params = {
            "engine": "openpyxl",
            "na_values": ["", "NA", "N/A", "null", "NULL", "none", "None"],
        }
        
        params = {**default_params, **kwargs}
        
        try:
            data = pd.read_excel(file_path, **params)
            return data
            
        except Exception as e:
            logger.error(f"Error loading Excel file {file_path}: {str(e)}")
            raise
    
    def _detect_csv_separator(self, file_path: str) -> str:
        """Detect CSV separator by examining the first few lines."""
        separators = [",", ";", "\t", "|"]
        
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                # Read first few lines
                lines = [file.readline().strip() for _ in range(3)]
            
            # Count occurrences of each separator
            separator_counts = {}
            for sep in separators:
                count = sum(line.count(sep) for line in lines if line)
                separator_counts[sep] = count
            
            # Return separator with highest count
            detected_sep = max(separator_counts, key=separator_counts.get)
            
            # If no clear winner, default to comma
            if separator_counts[detected_sep] == 0:
                detected_sep = ","
            
            logger.info(f"Detected CSV separator: '{detected_sep}'")
            return detected_sep
            
        except Exception:
            logger.warning("Could not detect CSV separator, defaulting to comma")
            return ","
    
    def _basic_csv_cleaning(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply basic cleaning to CSV data."""
        # Remove completely empty rows and columns
        data = data.dropna(how="all")  # Remove rows where all values are NaN
        data = data.dropna(axis=1, how="all")  # Remove columns where all values are NaN
        
        # Strip whitespace from string columns
        string_columns = data.select_dtypes(include=["object"]).columns
        for col in string_columns:
            data[col] = data[col].astype(str).str.strip()
            # Replace empty strings with NaN
            data[col] = data[col].replace("", np.nan)
        
        return data
    
    def register_loader(
        self,
        source_type: SourceType,
        loader_func: callable
    ) -> None:
        """
        Register a custom loader for a source type.
        
        Args:
            source_type: Type of data source
            loader_func: Function that takes (file_path, **kwargs) and returns DataFrame
        """
        self._loaders[source_type] = loader_func
        logger.info(f"Registered custom loader for source type: {source_type}")
    
    def get_supported_formats(self) -> list[SourceType]:
        """Get list of supported data formats."""
        return list(self._loaders.keys())
    
    def preview_data(
        self,
        source_path: str,
        source_type: Union[str, SourceType],
        n_rows: int = 5,
        **kwargs: Any
    ) -> pd.DataFrame:
        """
        Preview first few rows of data without loading the entire file.
        
        Args:
            source_path: Path to the data source
            source_type: Type of data source
            n_rows: Number of rows to preview
            **kwargs: Additional parameters for the loader
            
        Returns:
            Preview DataFrame with first n_rows
        """
        # Convert string to SourceType if needed
        if isinstance(source_type, str):
            source_type = SourceType(source_type.lower())
        
        # Add nrows parameter for supported formats
        if source_type == SourceType.CSV:
            kwargs["nrows"] = n_rows
        
        # Load data (with row limit if supported)
        data = self.load_data(source_path, source_type, **kwargs)
        
        # Return first n_rows
        return data.head(n_rows)
    
    def get_data_info(
        self,
        source_path: str,
        source_type: Union[str, SourceType],
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Get basic information about the data source.
        
        Args:
            source_path: Path to the data source
            source_type: Type of data source
            **kwargs: Additional parameters for the loader
            
        Returns:
            Dictionary with data information
        """
        # Convert string to SourceType if needed
        if isinstance(source_type, str):
            source_type = SourceType(source_type.lower())
        
        # Get file size
        file_path = Path(source_path)
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        
        # Preview data to get basic stats
        preview = self.preview_data(source_path, source_type, n_rows=100, **kwargs)
        
        info = {
            "file_path": str(file_path.absolute()),
            "file_size_mb": round(file_size_mb, 2),
            "source_type": source_type.value,
            "preview_shape": preview.shape,
            "columns": preview.columns.tolist(),
            "dtypes": preview.dtypes.to_dict(),
            "missing_values": preview.isnull().sum().to_dict(),
            "sample_values": {}
        }
        
        # Add sample values for each column
        for col in preview.columns:
            non_null_values = preview[col].dropna()
            if len(non_null_values) > 0:
                info["sample_values"][col] = non_null_values.iloc[:3].tolist()
            else:
                info["sample_values"][col] = []
        
        return info