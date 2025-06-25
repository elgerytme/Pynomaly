"""Polars data loader for high-performance data processing.

This module provides a Polars-based data loader that offers significant performance
improvements over pandas for large datasets through lazy evaluation and multi-threading.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pynomaly.domain.entities import Dataset
from pynomaly.domain.exceptions import DataLoadError
from pynomaly.shared.protocols import DataLoaderProtocol

logger = logging.getLogger(__name__)


class PolarsLoader(DataLoaderProtocol):
    """High-performance data loader using Polars with lazy evaluation."""

    def __init__(self, lazy: bool = True, streaming: bool = False):
        """Initialize Polars loader.

        Args:
            lazy: Whether to use lazy evaluation (recommended for large datasets)
            streaming: Enable streaming mode for extremely large datasets
        """
        self.lazy = lazy
        self.streaming = streaming
        self._validate_polars_availability()

    def _validate_polars_availability(self) -> None:
        """Validate that Polars is available."""
        try:
            import polars as pl

            self.pl = pl
            logger.info(f"Polars {pl.__version__} loaded successfully")
        except ImportError as e:
            raise DataLoadError(
                "Polars is required for PolarsLoader. Install with: pip install polars"
            ) from e

    async def load(self, file_path: str | Path, **kwargs) -> Dataset:
        """Load data using Polars with high-performance processing.

        Args:
            file_path: Path to the data file
            **kwargs: Additional loading parameters

        Returns:
            Dataset with Polars DataFrame converted to pandas for compatibility

        Raises:
            DataLoadError: If loading fails
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise DataLoadError(f"File not found: {file_path}")

        try:
            # Determine file type and load with appropriate method
            if file_path.suffix.lower() == ".csv":
                df = await self._load_csv(file_path, **kwargs)
            elif file_path.suffix.lower() in [".parquet", ".pq"]:
                df = await self._load_parquet(file_path, **kwargs)
            elif file_path.suffix.lower() in [".json", ".jsonl", ".ndjson"]:
                df = await self._load_json(file_path, **kwargs)
            elif file_path.suffix.lower() in [".xlsx", ".xls"]:
                df = await self._load_excel(file_path, **kwargs)
            else:
                raise DataLoadError(f"Unsupported file format: {file_path.suffix}")

            # Execute lazy frame if needed and convert to pandas for compatibility
            if hasattr(df, "collect"):
                if self.streaming:
                    # Process in chunks for streaming
                    df = self._process_streaming(df, **kwargs)
                else:
                    df = df.collect()

            # Convert to pandas DataFrame for compatibility with existing codebase
            pandas_df = df.to_pandas()

            # Extract metadata
            metadata = self._extract_metadata(df, file_path, **kwargs)

            # Create Dataset entity
            dataset = Dataset(
                name=kwargs.get("name", file_path.stem),
                data=pandas_df,
                description=kwargs.get("description"),
                file_path=str(file_path),
                target_column=kwargs.get("target_column"),
                features=list(pandas_df.columns),
                metadata=metadata,
            )

            logger.info(f"Successfully loaded {len(pandas_df)} rows from {file_path}")
            return dataset

        except Exception as e:
            logger.error(f"Failed to load data from {file_path}: {e}")
            raise DataLoadError(f"Failed to load data: {e}") from e

    async def _load_csv(self, file_path: Path, **kwargs) -> Any:
        """Load CSV file using Polars with optimized settings."""
        load_params = {
            "separator": kwargs.get("delimiter", ","),
            "has_header": kwargs.get("header", True),
            "skip_rows": kwargs.get("skiprows", 0),
            "n_rows": kwargs.get("nrows"),
            "encoding": kwargs.get("encoding", "utf-8"),
            "null_values": kwargs.get("na_values", ["", "NULL", "null", "NA", "na"]),
            "try_parse_dates": True,
            "rechunk": True,  # Optimize memory layout
        }

        # Filter out None values
        load_params = {k: v for k, v in load_params.items() if v is not None}

        if self.lazy:
            return self.pl.scan_csv(str(file_path), **load_params)
        else:
            return self.pl.read_csv(str(file_path), **load_params)

    async def _load_parquet(self, file_path: Path, **kwargs) -> Any:
        """Load Parquet file using Polars with optimized settings."""
        load_params = {
            "n_rows": kwargs.get("nrows"),
            "use_pyarrow": True,  # Use PyArrow engine for best compatibility
            "rechunk": True,
        }

        # Filter out None values
        load_params = {k: v for k, v in load_params.items() if v is not None}

        if self.lazy:
            return self.pl.scan_parquet(str(file_path), **load_params)
        else:
            return self.pl.read_parquet(str(file_path), **load_params)

    async def _load_json(self, file_path: Path, **kwargs) -> Any:
        """Load JSON file using Polars."""
        # JSON loading is typically eager in Polars
        return self.pl.read_json(str(file_path))

    async def _load_excel(self, file_path: Path, **kwargs) -> Any:
        """Load Excel file using Polars (via pandas for now)."""
        # Polars doesn't have native Excel support yet, fall back to pandas
        try:
            import pandas as pd

            pandas_df = pd.read_excel(
                file_path,
                sheet_name=kwargs.get("sheet_name", 0),
                header=kwargs.get("header", 0),
                nrows=kwargs.get("nrows"),
            )
            return self.pl.from_pandas(pandas_df)
        except ImportError as e:
            raise DataLoadError(
                "Pandas is required for Excel files. Install with: pip install pandas openpyxl"
            ) from e

    def _process_streaming(self, lazy_df: Any, **kwargs) -> Any:
        """Process large datasets in streaming mode."""
        batch_size = kwargs.get("batch_size", 10000)

        # For streaming, we collect in batches
        # This is a simplified implementation - real streaming would process incrementally
        logger.info(f"Processing in streaming mode with batch size {batch_size}")
        return lazy_df.collect()

    def _extract_metadata(self, df: Any, file_path: Path, **kwargs) -> dict[str, Any]:
        """Extract metadata from Polars DataFrame."""
        metadata = {
            "loader": "polars",
            "file_path": str(file_path),
            "file_size": file_path.stat().st_size,
            "n_rows": len(df) if hasattr(df, "__len__") else None,
            "n_columns": df.width if hasattr(df, "width") else len(df.columns),
            "column_names": df.columns,
            "dtypes": {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes, strict=False)},
            "lazy_evaluation": self.lazy,
            "streaming_mode": self.streaming,
        }

        # Add memory usage info if available
        if hasattr(df, "estimated_size"):
            metadata["estimated_size_bytes"] = df.estimated_size()

        # Add schema information
        if hasattr(df, "schema"):
            metadata["schema"] = {name: str(dtype) for name, dtype in df.schema.items()}

        return metadata

    async def validate_file(self, file_path: str | Path) -> bool:
        """Validate that file can be loaded with Polars.

        Args:
            file_path: Path to validate

        Returns:
            True if file can be loaded
        """
        file_path = Path(file_path)

        if not file_path.exists():
            return False

        # Check if file format is supported
        supported_formats = {
            ".csv",
            ".parquet",
            ".pq",
            ".json",
            ".jsonl",
            ".ndjson",
            ".xlsx",
            ".xls",
        }
        return file_path.suffix.lower() in supported_formats

    def get_supported_formats(self) -> list[str]:
        """Get list of supported file formats.

        Returns:
            List of supported file extensions
        """
        return [
            ".csv",
            ".parquet",
            ".pq",
            ".json",
            ".jsonl",
            ".ndjson",
            ".xlsx",
            ".xls",
        ]

    def get_performance_info(self) -> dict[str, Any]:
        """Get performance-related information about Polars setup.

        Returns:
            Dictionary with performance info
        """
        info = {
            "lazy_evaluation": self.lazy,
            "streaming_mode": self.streaming,
            "multithreading": True,  # Polars uses multithreading by default
        }

        try:
            # Get thread count if available
            info["thread_count"] = self.pl.thread_pool_size()
        except (AttributeError, Exception):
            pass

        return info


# Convenience function for quick loading
async def load_with_polars(
    file_path: str | Path, lazy: bool = True, streaming: bool = False, **kwargs
) -> Dataset:
    """Convenience function to load data with Polars.

    Args:
        file_path: Path to data file
        lazy: Use lazy evaluation
        streaming: Use streaming mode for large files
        **kwargs: Additional loading parameters

    Returns:
        Dataset object
    """
    loader = PolarsLoader(lazy=lazy, streaming=streaming)
    return await loader.load(file_path, **kwargs)


# Performance comparison utility
async def compare_performance(
    file_path: str | Path, compare_with_pandas: bool = True
) -> dict[str, Any]:
    """Compare loading performance between Polars and pandas.

    Args:
        file_path: Path to test file
        compare_with_pandas: Whether to include pandas comparison

    Returns:
        Performance comparison results
    """
    import time

    results = {}

    # Test Polars lazy loading
    start_time = time.time()
    polars_loader = PolarsLoader(lazy=True)
    polars_dataset = await polars_loader.load(file_path)
    polars_time = time.time() - start_time

    results["polars_lazy"] = {
        "time_seconds": polars_time,
        "rows": len(polars_dataset.data),
        "memory_usage": polars_dataset.data.memory_usage(deep=True).sum(),
    }

    # Test Polars eager loading
    start_time = time.time()
    polars_eager_loader = PolarsLoader(lazy=False)
    polars_eager_dataset = await polars_eager_loader.load(file_path)
    polars_eager_time = time.time() - start_time

    results["polars_eager"] = {
        "time_seconds": polars_eager_time,
        "rows": len(polars_eager_dataset.data),
        "memory_usage": polars_eager_dataset.data.memory_usage(deep=True).sum(),
    }

    # Compare with pandas if requested
    if compare_with_pandas:
        try:
            from .csv_loader import CSVLoader

            start_time = time.time()
            csv_loader = CSVLoader()
            pandas_dataset = await csv_loader.load(file_path)
            pandas_time = time.time() - start_time

            results["pandas"] = {
                "time_seconds": pandas_time,
                "rows": len(pandas_dataset.data),
                "memory_usage": pandas_dataset.data.memory_usage(deep=True).sum(),
            }

            # Calculate speedup
            results["speedup"] = {
                "polars_lazy_vs_pandas": pandas_time / polars_time,
                "polars_eager_vs_pandas": pandas_time / polars_eager_time,
            }

        except Exception as e:
            logger.warning(f"Could not compare with pandas: {e}")

    return results
