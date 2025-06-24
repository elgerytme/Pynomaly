"""Excel data loader implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import pandas as pd

from pynomaly.domain.entities import Dataset
from pynomaly.domain.exceptions import DataValidationError
from pynomaly.shared.protocols import BatchDataLoaderProtocol


class ExcelLoader(BatchDataLoaderProtocol):
    """Data loader for Excel files."""
    
    def __init__(
        self,
        sheet_name: Union[str, int, None] = 0,
        header: Union[int, List[int], None] = 0,
        skiprows: Optional[Union[int, List[int]]] = None
    ):
        """Initialize Excel loader.
        
        Args:
            sheet_name: Sheet name or index to load
            header: Row(s) to use as column headers
            skiprows: Rows to skip at the beginning
        """
        self.sheet_name = sheet_name
        self.header = header
        self.skiprows = skiprows
    
    @property
    def supported_formats(self) -> List[str]:
        """Get supported file formats."""
        return ["xlsx", "xls", "xlsm", "xlsb"]
    
    def load(
        self,
        source: Union[str, Path],
        name: Optional[str] = None,
        **kwargs: Any
    ) -> Dataset:
        """Load Excel file into a Dataset.
        
        Args:
            source: Path to Excel file
            name: Optional name for dataset
            **kwargs: Additional pandas read_excel arguments
            
        Returns:
            Loaded dataset
        """
        source_path = Path(source)
        
        if not self.validate(source_path):
            raise DataValidationError(
                f"Invalid Excel file: {source_path}",
                file_path=str(source_path)
            )
        
        # Prepare read options
        read_options = {
            "sheet_name": kwargs.get("sheet_name", self.sheet_name),
            "header": kwargs.get("header", self.header),
            "skiprows": kwargs.get("skiprows", self.skiprows),
            **{k: v for k, v in kwargs.items() if k not in ["sheet_name", "header", "skiprows", "target_column"]}
        }
        
        try:
            # Load data
            df = pd.read_excel(source_path, **read_options)
            
            # Handle empty dataframe
            if df.empty:
                raise DataValidationError(
                    "Excel file is empty",
                    file_path=str(source_path)
                )
            
            # Clean column names (remove unnamed columns, strip whitespace)
            df.columns = [
                col if not str(col).startswith('Unnamed') else f'Column_{i}'
                for i, col in enumerate(df.columns)
            ]
            df.columns = [str(col).strip() for col in df.columns]
            
            # Remove completely empty rows and columns
            df = df.dropna(how='all', axis=0)  # Remove empty rows
            df = df.dropna(how='all', axis=1)  # Remove empty columns
            
            if df.empty:
                raise DataValidationError(
                    "Excel file contains no data after cleaning",
                    file_path=str(source_path)
                )
            
            # Create dataset
            dataset_name = name or source_path.stem
            
            # Check for target column
            target_column = kwargs.get("target_column")
            if target_column and target_column not in df.columns:
                raise DataValidationError(
                    f"Target column '{target_column}' not found in Excel",
                    file_path=str(source_path),
                    available_columns=list(df.columns)
                )
            
            # Get sheet information for metadata
            sheet_info = self._get_sheet_info(source_path)
            
            dataset = Dataset(
                name=dataset_name,
                data=df,
                target_column=target_column,
                metadata={
                    "source": str(source_path),
                    "loader": "ExcelLoader",
                    "sheet_name": read_options["sheet_name"],
                    "sheets_available": sheet_info.get("sheet_names", []),
                    "file_size_mb": source_path.stat().st_size / 1024 / 1024,
                    "cleaned": True
                }
            )
            
            return dataset
            
        except FileNotFoundError:
            raise DataValidationError(
                f"Excel file not found: {source_path}",
                file_path=str(source_path)
            )
        except PermissionError:
            raise DataValidationError(
                f"Permission denied accessing Excel file: {source_path}",
                file_path=str(source_path)
            )
        except Exception as e:
            if "xlrd" in str(e) or "openpyxl" in str(e):
                raise DataValidationError(
                    f"Missing required Excel library. Install with: pip install openpyxl xlrd",
                    file_path=str(source_path)
                )
            raise DataValidationError(
                f"Failed to load Excel: {e}",
                file_path=str(source_path)
            ) from e
    
    def validate(self, source: Union[str, Path]) -> bool:
        """Validate if source is a valid Excel file.
        
        Args:
            source: Path to validate
            
        Returns:
            True if valid Excel source
        """
        source_path = Path(source)
        
        # Check if file exists
        if not source_path.exists() or not source_path.is_file():
            return False
        
        # Check extension
        valid_extensions = {".xlsx", ".xls", ".xlsm", ".xlsb"}
        if source_path.suffix.lower() not in valid_extensions:
            return False
        
        # Try to read Excel file metadata
        try:
            # Just try to get basic info without reading data
            excel_file = pd.ExcelFile(source_path)
            sheet_names = excel_file.sheet_names
            excel_file.close()
            
            return len(sheet_names) > 0
            
        except Exception:
            return False
    
    def load_batch(
        self,
        source: Union[str, Path],
        batch_size: int,
        name: Optional[str] = None,
        **kwargs: Any
    ) -> Iterator[Dataset]:
        """Load Excel in batches.
        
        Args:
            source: Path to Excel file
            batch_size: Number of rows per batch
            name: Optional name prefix
            **kwargs: Additional options
            
        Yields:
            Dataset batches
        """
        source_path = Path(source)
        
        if not self.validate(source_path):
            raise DataValidationError(
                f"Invalid Excel file: {source_path}",
                file_path=str(source_path)
            )
        
        # First, load the full dataset to determine its size
        full_dataset = self.load(source, name, **kwargs)
        df = full_dataset.data
        
        dataset_name = name or source_path.stem
        target_column = kwargs.get("target_column")
        
        # Split into batches
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i + batch_size].copy()
            
            if batch_df.empty:
                continue
            
            batch_dataset = Dataset(
                name=f"{dataset_name}_batch_{i // batch_size}",
                data=batch_df,
                target_column=target_column,
                metadata={
                    "source": str(source_path),
                    "loader": "ExcelLoader",
                    "batch_index": i // batch_size,
                    "batch_size": len(batch_df),
                    "is_batch": True,
                    "sheet_name": full_dataset.metadata.get("sheet_name"),
                    "parent_dataset": full_dataset.name
                }
            )
            
            yield batch_dataset
    
    def estimate_size(self, source: Union[str, Path]) -> Dict[str, Any]:
        """Estimate the size of the Excel file.
        
        Args:
            source: Path to Excel file
            
        Returns:
            Size information
        """
        source_path = Path(source)
        
        if not self.validate(source_path):
            raise DataValidationError(
                f"Invalid Excel file: {source_path}",
                file_path=str(source_path)
            )
        
        file_size_bytes = source_path.stat().st_size
        
        try:
            # Get basic file information
            excel_file = pd.ExcelFile(source_path)
            sheet_names = excel_file.sheet_names
            
            # Try to get info about the main sheet
            main_sheet = self.sheet_name if self.sheet_name is not None else 0
            if isinstance(main_sheet, int):
                main_sheet = sheet_names[main_sheet] if main_sheet < len(sheet_names) else sheet_names[0]
            
            # Read just the first few rows to get column info
            try:
                sample_df = pd.read_excel(
                    source_path,
                    sheet_name=main_sheet,
                    nrows=10,
                    header=self.header
                )
                
                n_columns = len(sample_df.columns)
                
                # Get the actual number of rows (this might be slow for large files)
                # For estimation, we'll read the entire sheet but just the shape
                try:
                    full_shape_df = pd.read_excel(
                        source_path,
                        sheet_name=main_sheet,
                        header=self.header,
                        usecols=[0]  # Just read first column to get row count
                    )
                    estimated_rows = len(full_shape_df)
                except Exception:
                    # Fallback estimation based on file size
                    # Rough estimate: 100 bytes per cell on average
                    estimated_rows = max(10, int(file_size_bytes / (n_columns * 100)))
                
                # Estimate memory usage
                numeric_columns = len(sample_df.select_dtypes(include=['number']).columns)
                text_columns = n_columns - numeric_columns
                
                # Rough estimate: 8 bytes per number, 50 bytes per text
                estimated_memory = (
                    estimated_rows * (numeric_columns * 8 + text_columns * 50)
                ) / 1024 / 1024
                
                excel_file.close()
                
                return {
                    "file_size_mb": file_size_bytes / 1024 / 1024,
                    "estimated_rows": estimated_rows,
                    "columns": n_columns,
                    "numeric_columns": numeric_columns,
                    "memory_usage_mb": estimated_memory,
                    "sheets_available": sheet_names,
                    "main_sheet": main_sheet,
                    "sample_dtypes": sample_df.dtypes.to_dict()
                }
                
            except Exception as e:
                excel_file.close()
                return {
                    "file_size_mb": file_size_bytes / 1024 / 1024,
                    "sheets_available": sheet_names,
                    "estimated_rows": "unknown",
                    "error": f"Could not read sheet data: {e}"
                }
            
        except Exception as e:
            return {
                "file_size_mb": file_size_bytes / 1024 / 1024,
                "estimated_rows": "unknown",
                "error": str(e)
            }
    
    def _get_sheet_info(self, source_path: Path) -> Dict[str, Any]:
        """Get information about available sheets."""
        
        try:
            excel_file = pd.ExcelFile(source_path)
            sheet_names = excel_file.sheet_names
            excel_file.close()
            
            return {
                "sheet_names": sheet_names,
                "sheet_count": len(sheet_names)
            }
        
        except Exception:
            return {
                "sheet_names": [],
                "sheet_count": 0
            }
    
    def list_sheets(self, source: Union[str, Path]) -> List[str]:
        """List all available sheets in the Excel file.
        
        Args:
            source: Path to Excel file
            
        Returns:
            List of sheet names
        """
        source_path = Path(source)
        
        if not self.validate(source_path):
            raise DataValidationError(
                f"Invalid Excel file: {source_path}",
                file_path=str(source_path)
            )
        
        try:
            excel_file = pd.ExcelFile(source_path)
            sheet_names = excel_file.sheet_names
            excel_file.close()
            return sheet_names
        
        except Exception as e:
            raise DataValidationError(
                f"Failed to read Excel sheets: {e}",
                file_path=str(source_path)
            ) from e
    
    def load_sheet(
        self,
        source: Union[str, Path],
        sheet_name: Union[str, int],
        name: Optional[str] = None,
        **kwargs: Any
    ) -> Dataset:
        """Load a specific sheet from Excel file.
        
        Args:
            source: Path to Excel file
            sheet_name: Name or index of sheet to load
            name: Optional name for dataset
            **kwargs: Additional options
            
        Returns:
            Dataset from specified sheet
        """
        kwargs["sheet_name"] = sheet_name
        dataset = self.load(source, name, **kwargs)
        
        # Update metadata to reflect specific sheet
        dataset.metadata["sheet_name"] = sheet_name
        if name is None:
            sheet_suffix = f"_sheet_{sheet_name}" if isinstance(sheet_name, int) else f"_{sheet_name}"
            dataset.name = f"{dataset.name}{sheet_suffix}"
        
        return dataset