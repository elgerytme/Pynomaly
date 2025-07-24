"""Data conversion service for format transformation and preprocessing."""

import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import structlog

from ..entities.dataset import Dataset, DatasetType

logger = structlog.get_logger()


class ConversionFormat(Enum):
    """Supported conversion formats."""
    CSV = "csv"
    JSON = "json"
    PARQUET = "parquet"
    PICKLE = "pickle"
    NUMPY = "numpy"
    XLSX = "xlsx"


class ConversionMethod(Enum):
    """Data conversion methods."""
    STANDARD = "standard"
    NORMALIZED = "normalized"
    STANDARDIZED = "standardized"
    SCALED = "scaled"


@dataclass
class ConversionConfig:
    """Configuration for data conversion."""
    input_format: ConversionFormat
    output_format: ConversionFormat
    method: ConversionMethod = ConversionMethod.STANDARD
    encoding: str = "utf-8"
    delimiter: str = ","
    normalize_columns: bool = True
    handle_missing: str = "drop"  # drop, fill, ignore
    fill_value: Any = None
    remove_duplicates: bool = True
    feature_scaling: bool = False
    datetime_format: Optional[str] = None


@dataclass
class ConversionResult:
    """Result of data conversion operation."""
    success: bool
    output_path: Optional[str]
    records_processed: int
    records_dropped: int
    columns_processed: int
    conversion_time: float
    warnings: List[str]
    errors: List[str]
    metadata: Dict[str, Any]


class DataConversionService:
    """Service for converting data between different formats and preprocessing."""
    
    def __init__(self):
        self.supported_formats = {
            ConversionFormat.CSV: [".csv"],
            ConversionFormat.JSON: [".json", ".jsonl"],
            ConversionFormat.PARQUET: [".parquet"],
            ConversionFormat.PICKLE: [".pkl", ".pickle"],
            ConversionFormat.NUMPY: [".npy", ".npz"],
            ConversionFormat.XLSX: [".xlsx", ".xls"]
        }
    
    async def convert_file(
        self,
        input_path: str,
        output_path: str,
        config: ConversionConfig
    ) -> ConversionResult:
        """Convert a file from one format to another."""
        start_time = datetime.now()
        warnings = []
        errors = []
        
        try:
            # Validate input file
            input_file = Path(input_path)
            if not input_file.exists():
                raise FileNotFoundError(f"Input file not found: {input_path}")
            
            # Load data
            logger.info("Loading input data", 
                       input_path=input_path,
                       input_format=config.input_format.value)
            
            data = await self._load_data(input_path, config, warnings, errors)
            
            if data is None or len(data) == 0:
                return ConversionResult(
                    success=False,
                    output_path=None,
                    records_processed=0,
                    records_dropped=0,
                    columns_processed=0,
                    conversion_time=0.0,
                    warnings=warnings,
                    errors=["No data loaded from input file"],
                    metadata={}
                )
            
            original_records = len(data)
            original_columns = len(data.columns) if hasattr(data, 'columns') else 0
            
            # Preprocess data
            data = await self._preprocess_data(data, config, warnings, errors)
            
            final_records = len(data) if data is not None else 0
            final_columns = len(data.columns) if hasattr(data, 'columns') else 0
            
            # Save data
            logger.info("Saving converted data",
                       output_path=output_path,
                       output_format=config.output_format.value)
            
            await self._save_data(data, output_path, config, warnings, errors)
            
            conversion_time = (datetime.now() - start_time).total_seconds()
            
            return ConversionResult(
                success=True,
                output_path=output_path,
                records_processed=final_records,
                records_dropped=original_records - final_records,
                columns_processed=final_columns,
                conversion_time=conversion_time,
                warnings=warnings,
                errors=errors,
                metadata={
                    "original_records": original_records,
                    "original_columns": original_columns,
                    "final_records": final_records,
                    "final_columns": final_columns,
                    "compression_ratio": final_records / original_records if original_records > 0 else 0
                }
            )
            
        except Exception as e:
            conversion_time = (datetime.now() - start_time).total_seconds()
            logger.error("Data conversion failed", 
                        input_path=input_path,
                        output_path=output_path,
                        error=str(e))
            
            return ConversionResult(
                success=False,
                output_path=None,
                records_processed=0,
                records_dropped=0,
                columns_processed=0,
                conversion_time=conversion_time,
                warnings=warnings,
                errors=errors + [str(e)],
                metadata={}
            )
    
    async def convert_dataset(
        self,
        dataset: Dataset,
        config: ConversionConfig
    ) -> pd.DataFrame:
        """Convert a Dataset object to DataFrame with preprocessing."""
        try:
            # Convert dataset to DataFrame
            data = dataset.to_dataframe()
            
            # Apply preprocessing
            return await self._preprocess_data(data, config, [], [])
            
        except Exception as e:
            logger.error("Dataset conversion failed", error=str(e))
            raise
    
    async def batch_convert(
        self,
        input_patterns: List[str],
        output_directory: str,
        config: ConversionConfig
    ) -> List[ConversionResult]:
        """Convert multiple files matching patterns."""
        results = []
        output_dir = Path(output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all matching files
        all_files = []
        for pattern in input_patterns:
            files = list(Path(".").glob(pattern))
            all_files.extend(files)
        
        logger.info("Starting batch conversion",
                   file_count=len(all_files),
                   output_directory=output_directory)
        
        for input_file in all_files:
            try:
                # Generate output filename
                output_name = self._generate_output_filename(
                    input_file.name,
                    config.output_format
                )
                output_path = output_dir / output_name
                
                # Convert file
                result = await self.convert_file(
                    str(input_file),
                    str(output_path),
                    config
                )
                results.append(result)
                
            except Exception as e:
                logger.error("Failed to convert file",
                           file=str(input_file),
                           error=str(e))
                
                results.append(ConversionResult(
                    success=False,
                    output_path=None,
                    records_processed=0,
                    records_dropped=0,
                    columns_processed=0,
                    conversion_time=0.0,
                    warnings=[],
                    errors=[str(e)],
                    metadata={"input_file": str(input_file)}
                ))
        
        return results
    
    async def _load_data(
        self,
        input_path: str,
        config: ConversionConfig,
        warnings: List[str],
        errors: List[str]
    ) -> Optional[pd.DataFrame]:
        """Load data from input file."""
        try:
            if config.input_format == ConversionFormat.CSV:
                return pd.read_csv(
                    input_path,
                    encoding=config.encoding,
                    delimiter=config.delimiter
                )
            
            elif config.input_format == ConversionFormat.JSON:
                return pd.read_json(input_path, encoding=config.encoding)
            
            elif config.input_format == ConversionFormat.PARQUET:
                return pd.read_parquet(input_path)
            
            elif config.input_format == ConversionFormat.PICKLE:
                return pd.read_pickle(input_path)
            
            elif config.input_format == ConversionFormat.XLSX:
                return pd.read_excel(input_path)
            
            elif config.input_format == ConversionFormat.NUMPY:
                if input_path.endswith('.npz'):
                    data = np.load(input_path)
                    # Use first array in npz file
                    array_name = list(data.keys())[0]
                    return pd.DataFrame(data[array_name])
                else:
                    array = np.load(input_path)
                    return pd.DataFrame(array)
            
            else:
                raise ValueError(f"Unsupported input format: {config.input_format}")
                
        except Exception as e:
            errors.append(f"Failed to load data: {str(e)}")
            return None
    
    async def _save_data(
        self,
        data: pd.DataFrame,
        output_path: str,
        config: ConversionConfig,
        warnings: List[str],
        errors: List[str]
    ):
        """Save data to output file."""
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            if config.output_format == ConversionFormat.CSV:
                data.to_csv(
                    output_path,
                    index=False,
                    encoding=config.encoding,
                    sep=config.delimiter
                )
            
            elif config.output_format == ConversionFormat.JSON:
                data.to_json(
                    output_path,
                    orient='records',
                    indent=2
                )
            
            elif config.output_format == ConversionFormat.PARQUET:
                data.to_parquet(output_path, index=False)
            
            elif config.output_format == ConversionFormat.PICKLE:
                data.to_pickle(output_path)
            
            elif config.output_format == ConversionFormat.XLSX:
                data.to_excel(output_path, index=False)
            
            elif config.output_format == ConversionFormat.NUMPY:
                if output_path.endswith('.npz'):
                    np.savez_compressed(output_path, data=data.values)
                else:
                    np.save(output_path, data.values)
            
            else:
                raise ValueError(f"Unsupported output format: {config.output_format}")
                
        except Exception as e:
            errors.append(f"Failed to save data: {str(e)}")
            raise
    
    async def _preprocess_data(
        self,
        data: pd.DataFrame,
        config: ConversionConfig,
        warnings: List[str],
        errors: List[str]
    ) -> pd.DataFrame:
        """Apply preprocessing steps to data."""
        try:
            original_shape = data.shape
            
            # Handle missing values
            if config.handle_missing == "drop":
                data = data.dropna()
                if data.shape[0] < original_shape[0]:
                    warnings.append(f"Dropped {original_shape[0] - data.shape[0]} rows with missing values")
            
            elif config.handle_missing == "fill":
                fill_value = config.fill_value if config.fill_value is not None else 0
                data = data.fillna(fill_value)
                warnings.append(f"Filled missing values with {fill_value}")
            
            # Remove duplicates
            if config.remove_duplicates:
                before_dedup = len(data)
                data = data.drop_duplicates()
                after_dedup = len(data)
                if before_dedup > after_dedup:
                    warnings.append(f"Removed {before_dedup - after_dedup} duplicate rows")
            
            # Normalize column names
            if config.normalize_columns:
                original_columns = data.columns.tolist()
                data.columns = [
                    col.lower().replace(' ', '_').replace('-', '_') 
                    for col in data.columns
                ]
                if data.columns.tolist() != original_columns:
                    warnings.append("Column names normalized")
            
            # Apply scaling/normalization
            if config.method != ConversionMethod.STANDARD:
                data = await self._apply_scaling(data, config.method, warnings)
            
            # Feature scaling for numeric columns
            if config.feature_scaling:
                numeric_columns = data.select_dtypes(include=[np.number]).columns
                if len(numeric_columns) > 0:
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
                    warnings.append(f"Applied feature scaling to {len(numeric_columns)} numeric columns")
            
            return data
            
        except Exception as e:
            errors.append(f"Preprocessing failed: {str(e)}")
            return data
    
    async def _apply_scaling(
        self,
        data: pd.DataFrame,
        method: ConversionMethod,
        warnings: List[str]
    ) -> pd.DataFrame:
        """Apply scaling method to numeric columns."""
        try:
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            
            if len(numeric_columns) == 0:
                return data
            
            if method == ConversionMethod.NORMALIZED:
                # Min-max normalization
                for col in numeric_columns:
                    min_val = data[col].min()
                    max_val = data[col].max()
                    if max_val > min_val:
                        data[col] = (data[col] - min_val) / (max_val - min_val)
                warnings.append(f"Applied min-max normalization to {len(numeric_columns)} columns")
            
            elif method == ConversionMethod.STANDARDIZED:
                # Z-score standardization
                for col in numeric_columns:
                    mean_val = data[col].mean()
                    std_val = data[col].std()
                    if std_val > 0:
                        data[col] = (data[col] - mean_val) / std_val
                warnings.append(f"Applied z-score standardization to {len(numeric_columns)} columns")
            
            elif method == ConversionMethod.SCALED:
                # Robust scaling
                for col in numeric_columns:
                    median_val = data[col].median()
                    mad_val = data[col].mad()  # Median absolute deviation
                    if mad_val > 0:
                        data[col] = (data[col] - median_val) / mad_val
                warnings.append(f"Applied robust scaling to {len(numeric_columns)} columns")
            
            return data
            
        except Exception as e:
            warnings.append(f"Scaling failed: {str(e)}")
            return data
    
    def _generate_output_filename(self, input_name: str, output_format: ConversionFormat) -> str:
        """Generate output filename based on input and format."""
        input_path = Path(input_name)
        stem = input_path.stem
        
        extensions = {
            ConversionFormat.CSV: ".csv",
            ConversionFormat.JSON: ".json",
            ConversionFormat.PARQUET: ".parquet",
            ConversionFormat.PICKLE: ".pkl",
            ConversionFormat.NUMPY: ".npy",
            ConversionFormat.XLSX: ".xlsx"
        }
        
        extension = extensions.get(output_format, ".dat")
        return f"{stem}{extension}"
    
    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Get supported input/output formats."""
        return {
            format_enum.value: extensions
            for format_enum, extensions in self.supported_formats.items()
        }
    
    async def validate_conversion_config(self, config: ConversionConfig) -> Tuple[bool, List[str]]:
        """Validate conversion configuration."""
        errors = []
        
        # Check format support
        if config.input_format not in self.supported_formats:
            errors.append(f"Unsupported input format: {config.input_format}")
        
        if config.output_format not in self.supported_formats:
            errors.append(f"Unsupported output format: {config.output_format}")
        
        # Check handle_missing value
        if config.handle_missing not in ["drop", "fill", "ignore"]:
            errors.append("handle_missing must be one of: drop, fill, ignore")
        
        # Check fill_value when using fill method
        if config.handle_missing == "fill" and config.fill_value is None:
            errors.append("fill_value must be provided when handle_missing is 'fill'")
        
        return len(errors) == 0, errors
    
    async def get_conversion_statistics(self, results: List[ConversionResult]) -> Dict[str, Any]:
        """Generate statistics from conversion results."""
        if not results:
            return {}
        
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        total_records = sum(r.records_processed for r in successful)
        total_dropped = sum(r.records_dropped for r in successful)
        total_time = sum(r.conversion_time for r in results)
        
        return {
            "total_conversions": len(results),
            "successful_conversions": len(successful),
            "failed_conversions": len(failed),
            "success_rate": len(successful) / len(results) * 100 if results else 0,
            "total_records_processed": total_records,
            "total_records_dropped": total_dropped,
            "drop_rate": total_dropped / (total_records + total_dropped) * 100 if (total_records + total_dropped) > 0 else 0,
            "total_conversion_time": total_time,
            "average_conversion_time": total_time / len(results) if results else 0,
            "throughput": total_records / total_time if total_time > 0 else 0
        }