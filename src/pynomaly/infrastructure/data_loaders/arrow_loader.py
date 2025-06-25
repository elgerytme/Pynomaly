"""Native Apache Arrow data loader with compute functions and streaming support.

This module provides comprehensive Arrow-native data processing capabilities,
including streaming, compute functions, and columnar operations optimization.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pynomaly.domain.entities import Dataset
from pynomaly.domain.exceptions import DataLoadError
from pynomaly.shared.protocols import DataLoaderProtocol

logger = logging.getLogger(__name__)


class ArrowLoader(DataLoaderProtocol):
    """Native Apache Arrow data loader with advanced processing capabilities."""

    def __init__(self, use_threads: bool = True, memory_pool: Any | None = None):
        """Initialize Arrow loader.

        Args:
            use_threads: Whether to use multithreading for operations
            memory_pool: Custom memory pool for Arrow operations
        """
        self.use_threads = use_threads
        self.memory_pool = memory_pool
        self._validate_arrow_availability()

    def _validate_arrow_availability(self) -> None:
        """Validate that PyArrow is available with required features."""
        try:
            import pyarrow as pa
            import pyarrow.compute as pc
            import pyarrow.csv as csv
            import pyarrow.dataset as ds
            import pyarrow.json as json
            import pyarrow.parquet as pq

            self.pa = pa
            self.pc = pc
            self.ds = ds
            self.pq = pq
            self.csv = csv
            self.json = json

            logger.info(f"PyArrow {pa.__version__} loaded with compute engine")

        except ImportError as e:
            raise DataLoadError(
                "PyArrow with compute functions is required. Install with: pip install pyarrow"
            ) from e

    async def load(self, file_path: str | Path, **kwargs) -> Dataset:
        """Load data using native Arrow with columnar optimizations.

        Args:
            file_path: Path to the data file or directory
            **kwargs: Additional loading parameters

        Returns:
            Dataset with Arrow table converted to pandas for compatibility

        Raises:
            DataLoadError: If loading fails
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise DataLoadError(f"Path not found: {file_path}")

        try:
            # Load as Arrow table
            if file_path.is_dir():
                # Dataset directory
                table = await self._load_dataset(file_path, **kwargs)
            else:
                # Single file
                table = await self._load_file(file_path, **kwargs)

            # Apply Arrow compute functions if specified
            table = await self._apply_transforms(table, **kwargs)

            # Convert to pandas for compatibility (with optimizations)
            pandas_df = self._arrow_to_pandas_optimized(table, **kwargs)

            # Extract metadata
            metadata = self._extract_metadata(table, file_path, **kwargs)

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

    async def _load_file(self, file_path: Path, **kwargs) -> Any:
        """Load single file as Arrow table."""
        suffix = file_path.suffix.lower()

        if suffix in [".parquet", ".pq"]:
            return await self._load_parquet_native(file_path, **kwargs)
        elif suffix == ".csv":
            return await self._load_csv_native(file_path, **kwargs)
        elif suffix in [".json", ".jsonl", ".ndjson"]:
            return await self._load_json_native(file_path, **kwargs)
        elif suffix in [".arrow", ".feather"]:
            return await self._load_arrow_native(file_path, **kwargs)
        else:
            raise DataLoadError(f"Unsupported format for native Arrow: {suffix}")

    async def _load_dataset(self, dir_path: Path, **kwargs) -> Any:
        """Load dataset directory as Arrow table."""
        # Determine dataset format
        format_type = kwargs.get("format", "parquet")

        if format_type == "parquet":
            dataset = self.ds.dataset(str(dir_path), format="parquet")
        elif format_type == "csv":
            dataset = self.ds.dataset(str(dir_path), format="csv")
        else:
            # Auto-detect format
            dataset = self.ds.dataset(str(dir_path))

        # Apply filters if specified
        filters = kwargs.get("filters")
        columns = kwargs.get("columns")

        # Scan with optional filters and column selection
        scan_builder = dataset.scanner(
            columns=columns, filter=filters, use_threads=self.use_threads
        )

        return scan_builder.to_table()

    async def _load_parquet_native(self, file_path: Path, **kwargs) -> Any:
        """Load Parquet file with native Arrow optimizations."""
        read_options = {
            "use_threads": self.use_threads,
            "memory_map": kwargs.get("memory_map", True),
        }

        # Column selection for better performance
        columns = kwargs.get("columns")
        if columns:
            read_options["columns"] = columns

        # Row group filters
        filters = kwargs.get("filters")

        # Use ParquetFile for more control
        parquet_file = self.pq.ParquetFile(str(file_path))

        if filters or kwargs.get("row_groups"):
            # Selective reading
            row_groups = kwargs.get("row_groups")
            return parquet_file.read(
                columns=columns,
                row_groups=row_groups,
                filters=filters,
                use_threads=self.use_threads,
            )
        else:
            # Full read
            return parquet_file.read(**read_options)

    async def _load_csv_native(self, file_path: Path, **kwargs) -> Any:
        """Load CSV file with native Arrow CSV reader."""
        # Configure CSV parsing options
        parse_options = self.csv.ParseOptions(
            delimiter=kwargs.get("delimiter", ","),
            quote_char=kwargs.get("quote_char", '"'),
            escape_char=kwargs.get("escape_char"),
            newlines_in_values=kwargs.get("newlines_in_values", False),
        )

        # Configure reading options
        read_options = self.csv.ReadOptions(
            use_threads=self.use_threads,
            block_size=kwargs.get("block_size", 1024 * 1024),  # 1MB blocks
            skip_rows=kwargs.get("skip_rows", 0),
            column_names=kwargs.get("column_names"),
            encoding=kwargs.get("encoding", "utf8"),
        )

        # Configure conversion options
        convert_options = self.csv.ConvertOptions(
            check_utf8=kwargs.get("check_utf8", True),
            column_types=kwargs.get("column_types"),
            null_values=kwargs.get("null_values", ["", "NULL", "null"]),
            true_values=kwargs.get("true_values", ["true", "True", "TRUE"]),
            false_values=kwargs.get("false_values", ["false", "False", "FALSE"]),
            timestamp_parsers=kwargs.get("timestamp_parsers"),
        )

        return self.csv.read_csv(
            str(file_path),
            parse_options=parse_options,
            read_options=read_options,
            convert_options=convert_options,
        )

    async def _load_json_native(self, file_path: Path, **kwargs) -> Any:
        """Load JSON file with native Arrow JSON reader."""
        read_options = self.json.ReadOptions(
            use_threads=self.use_threads,
            block_size=kwargs.get("block_size", 1024 * 1024),
        )

        parse_options = self.json.ParseOptions(
            explicit_schema=kwargs.get("schema"),
            newlines_in_values=kwargs.get("newlines_in_values", False),
        )

        return self.json.read_json(
            str(file_path), read_options=read_options, parse_options=parse_options
        )

    async def _load_arrow_native(self, file_path: Path, **kwargs) -> Any:
        """Load native Arrow/Feather file."""
        if file_path.suffix.lower() == ".feather":
            # Use feather reader
            import pyarrow.feather as feather

            return feather.read_table(str(file_path))
        else:
            # Use Arrow IPC reader
            with self.pa.ipc.open_file(str(file_path)) as reader:
                return reader.read_all()

    async def _apply_transforms(self, table: Any, **kwargs) -> Any:
        """Apply Arrow compute functions and transformations."""
        if not kwargs.get("transforms"):
            return table

        transforms = kwargs["transforms"]

        for transform in transforms:
            transform_type = transform.get("type")

            if transform_type == "filter":
                # Apply filter using compute functions
                condition = transform["condition"]
                table = table.filter(condition)

            elif transform_type == "select":
                # Select specific columns
                columns = transform["columns"]
                table = table.select(columns)

            elif transform_type == "compute":
                # Apply compute function
                operation = transform["operation"]
                column = transform["column"]
                new_column = transform.get("new_column", f"{column}_{operation}")

                if operation == "normalize":
                    # Min-max normalization
                    col_data = table[column]
                    min_val = self.pc.min(col_data)
                    max_val = self.pc.max(col_data)
                    normalized = self.pc.divide(
                        self.pc.subtract(col_data, min_val),
                        self.pc.subtract(max_val, min_val),
                    )
                    table = table.add_column(table.num_columns, new_column, normalized)

                elif operation == "zscore":
                    # Z-score normalization
                    col_data = table[column]
                    mean_val = self.pc.mean(col_data)
                    std_val = self.pc.stddev(col_data)
                    zscore = self.pc.divide(
                        self.pc.subtract(col_data, mean_val), std_val
                    )
                    table = table.add_column(table.num_columns, new_column, zscore)

                elif operation == "log":
                    # Log transformation
                    col_data = table[column]
                    log_data = self.pc.ln(col_data)
                    table = table.add_column(table.num_columns, new_column, log_data)

            elif transform_type == "aggregate":
                # Group by aggregation
                transform["group_by"]
                transform["aggregates"]

                # This would require more complex groupby logic
                logger.warning(
                    "Group by aggregation not fully implemented in this version"
                )

        return table

    def _arrow_to_pandas_optimized(self, table: Any, **kwargs) -> Any:
        """Convert Arrow table to pandas with optimizations."""
        convert_options = {
            "use_threads": self.use_threads,
            "zero_copy_only": kwargs.get("zero_copy_only", False),
            "integer_object_nulls": kwargs.get("integer_object_nulls", False),
        }

        # Handle categorical columns
        if kwargs.get("categories"):
            convert_options["categories"] = kwargs["categories"]

        return table.to_pandas(**convert_options)

    def _extract_metadata(
        self, table: Any, file_path: Path, **kwargs
    ) -> dict[str, Any]:
        """Extract metadata from Arrow table."""
        metadata = {
            "loader": "arrow_native",
            "file_path": str(file_path),
            "n_rows": len(table),
            "n_columns": table.num_columns,
            "column_names": table.column_names,
            "schema": str(table.schema),
            "arrow_version": self.pa.__version__,
            "use_threads": self.use_threads,
        }

        # Add file size if it's a file
        if file_path.is_file():
            metadata["file_size"] = file_path.stat().st_size

        # Add memory usage
        try:
            metadata["memory_usage_bytes"] = table.nbytes
        except AttributeError:
            pass

        # Add column types
        metadata["column_types"] = {
            name: str(field.type)
            for name, field in zip(table.column_names, table.schema, strict=False)
        }

        # Add null counts
        metadata["null_counts"] = {
            col_name: self.pc.count(table[col_name], mode="only_null").as_py()
            for col_name in table.column_names
        }

        return metadata

    async def load_streaming(
        self, file_path: str | Path, batch_size: int = 10000, **kwargs
    ) -> Iterator[Dataset]:
        """Load data in streaming mode for large files.

        Args:
            file_path: Path to data file
            batch_size: Number of rows per batch
            **kwargs: Additional parameters

        Yields:
            Dataset objects for each batch
        """
        file_path = Path(file_path)

        try:
            if file_path.suffix.lower() in [".parquet", ".pq"]:
                async for batch in self._stream_parquet(
                    file_path, batch_size, **kwargs
                ):
                    yield batch
            elif file_path.suffix.lower() == ".csv":
                async for batch in self._stream_csv(file_path, batch_size, **kwargs):
                    yield batch
            else:
                raise DataLoadError(f"Streaming not supported for {file_path.suffix}")

        except Exception as e:
            logger.error(f"Streaming failed for {file_path}: {e}")
            raise DataLoadError(f"Streaming failed: {e}") from e

    async def _stream_parquet(
        self, file_path: Path, batch_size: int, **kwargs
    ) -> Iterator[Dataset]:
        """Stream Parquet file in batches."""
        parquet_file = self.pq.ParquetFile(str(file_path))

        # Read row groups as batches
        for row_group_idx in range(parquet_file.num_row_groups):
            table = parquet_file.read_row_group(
                row_group_idx,
                columns=kwargs.get("columns"),
                use_threads=self.use_threads,
            )

            # Convert to pandas
            pandas_df = self._arrow_to_pandas_optimized(table, **kwargs)

            # Create dataset
            dataset = Dataset(
                name=f"{file_path.stem}_batch_{row_group_idx}",
                data=pandas_df,
                file_path=str(file_path),
                target_column=kwargs.get("target_column"),
                features=list(pandas_df.columns),
                metadata={
                    "batch_idx": row_group_idx,
                    "batch_size": len(pandas_df),
                    "is_streaming_batch": True,
                },
            )

            yield dataset

    async def _stream_csv(
        self, file_path: Path, batch_size: int, **kwargs
    ) -> Iterator[Dataset]:
        """Stream CSV file in batches."""
        # For CSV streaming, we need to use a different approach
        # This is a simplified implementation
        total_table = await self._load_csv_native(file_path, **kwargs)

        # Split into batches
        total_rows = len(total_table)
        for start_idx in range(0, total_rows, batch_size):
            end_idx = min(start_idx + batch_size, total_rows)
            batch_table = total_table.slice(start_idx, end_idx - start_idx)

            # Convert to pandas
            pandas_df = self._arrow_to_pandas_optimized(batch_table, **kwargs)

            # Create dataset
            dataset = Dataset(
                name=f"{file_path.stem}_batch_{start_idx // batch_size}",
                data=pandas_df,
                file_path=str(file_path),
                target_column=kwargs.get("target_column"),
                features=list(pandas_df.columns),
                metadata={
                    "batch_idx": start_idx // batch_size,
                    "batch_size": len(pandas_df),
                    "start_row": start_idx,
                    "end_row": end_idx,
                    "is_streaming_batch": True,
                },
            )

            yield dataset

    async def validate_file(self, file_path: str | Path) -> bool:
        """Validate that file can be loaded with native Arrow.

        Args:
            file_path: Path to validate

        Returns:
            True if file can be loaded
        """
        file_path = Path(file_path)

        if not file_path.exists():
            return False

        # Check supported formats
        supported_formats = {
            ".parquet",
            ".pq",
            ".csv",
            ".json",
            ".jsonl",
            ".ndjson",
            ".arrow",
            ".feather",
        }
        return file_path.suffix.lower() in supported_formats

    def get_supported_formats(self) -> list[str]:
        """Get list of supported file formats."""
        return [
            ".parquet",
            ".pq",
            ".csv",
            ".json",
            ".jsonl",
            ".ndjson",
            ".arrow",
            ".feather",
        ]

    def get_compute_functions(self) -> list[str]:
        """Get list of available Arrow compute functions."""
        # Return commonly used compute functions
        return [
            "add",
            "subtract",
            "multiply",
            "divide",
            "min",
            "max",
            "mean",
            "sum",
            "count",
            "stddev",
            "variance",
            "ln",
            "log10",
            "abs",
            "ceil",
            "floor",
            "round",
            "is_null",
            "is_valid",
            "fill_null",
            "greater",
            "less",
            "equal",
            "not_equal",
            "and",
            "or",
            "not",
            "if_else",
        ]


# Convenience functions
async def load_with_arrow(
    file_path: str | Path, use_threads: bool = True, **kwargs
) -> Dataset:
    """Convenience function to load data with native Arrow.

    Args:
        file_path: Path to data file
        use_threads: Use multithreading
        **kwargs: Additional parameters

    Returns:
        Dataset object
    """
    loader = ArrowLoader(use_threads=use_threads)
    return await loader.load(file_path, **kwargs)


async def stream_with_arrow(
    file_path: str | Path, batch_size: int = 10000, **kwargs
) -> Iterator[Dataset]:
    """Convenience function to stream data with native Arrow.

    Args:
        file_path: Path to data file
        batch_size: Batch size for streaming
        **kwargs: Additional parameters

    Yields:
        Dataset objects for each batch
    """
    loader = ArrowLoader()
    async for batch in loader.load_streaming(file_path, batch_size, **kwargs):
        yield batch
