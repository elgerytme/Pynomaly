"""Enhanced data loader factory with integrated data transformation capabilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from pynomaly.domain.entities import Dataset
from pynomaly.domain.exceptions import DataValidationError
from pynomaly.shared.protocols import DataLoaderProtocol

# Import existing loaders
from .data_loader_factory import DataLoaderFactory
from .csv_loader import CSVLoader
from .excel_loader import ExcelLoader
from .json_loader import JSONLoader
from .parquet_loader import ParquetLoader

# Import data_transformation components
try:
    from data_transformation.application.use_cases.data_pipeline import DataPipelineUseCase
    from data_transformation.domain.value_objects.pipeline_config import (
        PipelineConfig, SourceType, CleaningStrategy, ScalingMethod, EncodingStrategy
    )
    from data_transformation.infrastructure.adapters.data_source_adapter import DataSourceAdapter
    DATA_TRANSFORMATION_AVAILABLE = True
except ImportError:
    DATA_TRANSFORMATION_AVAILABLE = False


class EnhancedDataLoaderFactory(DataLoaderFactory):
    """Enhanced data loader factory with integrated preprocessing capabilities."""

    def __init__(self, enable_auto_preprocessing: bool = True):
        """Initialize the enhanced data loader factory.
        
        Args:
            enable_auto_preprocessing: Whether to enable automatic data preprocessing
        """
        super().__init__()
        self.enable_auto_preprocessing = enable_auto_preprocessing
        self.logger = logging.getLogger(__name__)
        
        if DATA_TRANSFORMATION_AVAILABLE:
            self.transformation_adapter = DataSourceAdapter()
            self.logger.info("Data transformation capabilities enabled")
        else:
            self.logger.warning("Data transformation package not available")

    def load_and_transform(
        self,
        source: str | Path,
        transformation_config: Optional[PipelineConfig] = None,
        auto_detect_issues: bool = True,
        **loader_kwargs: Any
    ) -> pd.DataFrame:
        """Load data and apply intelligent transformations.
        
        Args:
            source: Data source path or identifier
            transformation_config: Optional transformation configuration
            auto_detect_issues: Whether to automatically detect and fix data quality issues
            **loader_kwargs: Additional arguments for the data loader
            
        Returns:
            Transformed DataFrame ready for anomaly detection
            
        Raises:
            DataValidationError: If data loading or transformation fails
        """
        try:
            # First, load the data using existing loaders
            raw_data = self.load_data(source, **loader_kwargs)
            
            if not DATA_TRANSFORMATION_AVAILABLE or not self.enable_auto_preprocessing:
                return raw_data
            
            # Create transformation configuration if not provided
            if transformation_config is None:
                transformation_config = self._create_optimal_config(
                    source, raw_data, auto_detect_issues
                )
            
            # Apply data transformation pipeline
            pipeline = DataPipelineUseCase(transformation_config)
            result = pipeline.execute(raw_data)
            
            if result.success:
                self.logger.info(
                    f"Successfully transformed data: {len(result.data)} rows, "
                    f"{len(result.data.columns)} columns"
                )
                return result.data
            else:
                self.logger.warning(
                    f"Data transformation failed: {result.error_message}. "
                    "Returning raw data."
                )
                return raw_data
                
        except Exception as e:
            self.logger.error(f"Error in load_and_transform: {e}")
            raise DataValidationError(f"Failed to load and transform data: {e}")

    def get_transformation_recommendations(
        self,
        source: str | Path,
        **loader_kwargs: Any
    ) -> dict[str, Any]:
        """Get intelligent preprocessing recommendations for the dataset.
        
        Args:
            source: Data source path or identifier
            **loader_kwargs: Additional arguments for the data loader
            
        Returns:
            Dictionary containing transformation recommendations
        """
        if not DATA_TRANSFORMATION_AVAILABLE:
            return {"error": "Data transformation package not available"}
        
        try:
            # Load a sample of the data for analysis
            raw_data = self.load_data(source, **loader_kwargs)
            
            # Use transformation adapter to analyze data quality
            source_type = self._detect_source_type(source)
            validation_result = self.transformation_adapter.validate_source(str(source), source_type)
            
            # Generate recommendations based on data analysis
            recommendations = {
                "data_quality": {
                    "is_valid": validation_result[0],
                    "issues": validation_result[1] if not validation_result[0] else []
                },
                "suggested_config": self._create_optimal_config(source, raw_data, True),
                "preprocessing_steps": self._recommend_preprocessing_steps(raw_data),
                "data_summary": {
                    "rows": len(raw_data),
                    "columns": len(raw_data.columns),
                    "missing_values": raw_data.isnull().sum().sum(),
                    "numeric_columns": len(raw_data.select_dtypes(include=['number']).columns),
                    "categorical_columns": len(raw_data.select_dtypes(include=['object']).columns)
                }
            }
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return {"error": f"Failed to generate recommendations: {e}"}

    def _create_optimal_config(
        self,
        source: str | Path,
        data: pd.DataFrame,
        auto_detect_issues: bool
    ) -> PipelineConfig:
        """Create optimal transformation configuration based on data analysis."""
        source_type = self._detect_source_type(source)
        
        # Analyze data characteristics
        missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
        categorical_ratio = len(data.select_dtypes(include=['object']).columns) / len(data.columns)
        
        # Choose appropriate strategies based on data characteristics
        if missing_ratio > 0.1:
            cleaning_strategy = CleaningStrategy.STATISTICAL
        elif auto_detect_issues:
            cleaning_strategy = CleaningStrategy.AUTO
        else:
            cleaning_strategy = CleaningStrategy.CONSERVATIVE
            
        if categorical_ratio > 0.5:
            encoding_strategy = EncodingStrategy.ONEHOT
        else:
            encoding_strategy = EncodingStrategy.LABEL
        
        return PipelineConfig(
            source_type=source_type,
            cleaning_strategy=cleaning_strategy,
            scaling_method=ScalingMethod.ROBUST,  # Robust to outliers
            encoding_strategy=encoding_strategy,
            feature_engineering=True,
            validation_enabled=True,
            parallel_processing=True
        )

    def _detect_source_type(self, source: str | Path) -> SourceType:
        """Detect source type from file extension or path."""
        if isinstance(source, Path):
            source_str = str(source)
        else:
            source_str = source
            
        extension = Path(source_str).suffix.lower()
        
        mapping = {
            '.csv': SourceType.CSV,
            '.tsv': SourceType.CSV,
            '.txt': SourceType.CSV,
            '.json': SourceType.JSON,
            '.jsonl': SourceType.JSON,
            '.parquet': SourceType.PARQUET,
            '.pq': SourceType.PARQUET,
            '.xlsx': SourceType.EXCEL,
            '.xls': SourceType.EXCEL
        }
        
        return mapping.get(extension, SourceType.CSV)

    def _recommend_preprocessing_steps(self, data: pd.DataFrame) -> list[str]:
        """Recommend specific preprocessing steps based on data analysis."""
        recommendations = []
        
        # Check for missing values
        if data.isnull().any().any():
            recommendations.append("Handle missing values")
            
        # Check for duplicates
        if data.duplicated().any():
            recommendations.append("Remove duplicate rows")
            
        # Check for outliers in numeric columns
        numeric_cols = data.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            q1 = data[col].quantile(0.25)
            q3 = data[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
            if outliers > 0:
                recommendations.append(f"Handle outliers in {col} ({outliers} detected)")
                
        # Check for categorical variables that need encoding
        categorical_cols = data.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            recommendations.append(f"Encode categorical variables ({len(categorical_cols)} columns)")
            
        # Check for feature scaling needs
        if len(numeric_cols) > 1:
            # Check if features have different scales
            scales = data[numeric_cols].std()
            if (scales.max() / scales.min()) > 10:
                recommendations.append("Scale features (different magnitudes detected)")
                
        return recommendations