"""Parquet data loader implementation."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from pynomaly.domain.entities import Dataset
from pynomaly.domain.exceptions import DataValidationError
from pynomaly.shared.protocols import BatchDataLoaderProtocol


class ParquetLoader(BatchDataLoaderProtocol):
    """Data loader for Parquet files."""

    def __init__(self, use_pyarrow: bool = True, columns: list[str] | None = None):
        """Initialize Parquet loader.

        Args:
            use_pyarrow: Whether to use PyArrow engine
            columns: Specific columns to load
        """
        self.use_pyarrow = use_pyarrow
        self.columns = columns

    @property
    def supported_formats(self) -> list[str]:
        """Get supported file formats."""
        return ["parquet", "pq"]

    def load(
        self, source: str | Path, name: str | None = None, **kwargs: Any
    ) -> Dataset:
        """Load Parquet file into a Dataset.

        Args:
            source: Path to Parquet file
            name: Optional name for dataset
            **kwargs: Additional arguments

        Returns:
            Loaded dataset
        """
        source_path = Path(source)

        if not self.validate(source_path):
            raise DataValidationError(
                f"Invalid Parquet file: {source_path}", file_path=str(source_path)
            )

        try:
            # Prepare read options
            columns_to_read = kwargs.get("columns", self.columns)

            if self.use_pyarrow:
                # Use PyArrow for better performance
                df = pd.read_parquet(
                    source_path, engine="pyarrow", columns=columns_to_read
                )
            else:
                # Use fastparquet as alternative
                df = pd.read_parquet(
                    source_path, engine="fastparquet", columns=columns_to_read
                )

            # Handle empty dataframe
            if df.empty:
                raise DataValidationError(
                    "Parquet file is empty", file_path=str(source_path)
                )

            # Create dataset
            dataset_name = name or source_path.stem

            # Check for target column
            target_column = kwargs.get("target_column")
            if target_column and target_column not in df.columns:
                raise DataValidationError(
                    f"Target column '{target_column}' not found in Parquet file",
                    file_path=str(source_path),
                    available_columns=list(df.columns),
                )

            # Get Parquet metadata
            parquet_file = pq.ParquetFile(source_path)
            metadata = {
                "source": str(source_path),
                "loader": "ParquetLoader",
                "file_size_mb": source_path.stat().st_size / 1024 / 1024,
                "num_row_groups": parquet_file.num_row_groups,
                "compression": str(
                    parquet_file.metadata.row_group(0).column(0).compression
                ),
                "created_by": str(parquet_file.metadata.created_by)
                if parquet_file.metadata.created_by
                else None,
            }

            dataset = Dataset(
                name=dataset_name,
                data=df,
                target_column=target_column,
                metadata=metadata,
            )

            return dataset

        except Exception as e:
            raise DataValidationError(
                f"Failed to load Parquet file: {e}", file_path=str(source_path)
            ) from e

    def validate(self, source: str | Path) -> bool:
        """Validate if source is a valid Parquet file.

        Args:
            source: Path to validate

        Returns:
            True if valid Parquet source
        """
        source_path = Path(source)

        # Check if file exists
        if not source_path.exists():
            return False

        # Check if it's a file
        if not source_path.is_file():
            return False

        # Check extension
        valid_extensions = {".parquet", ".pq"}
        if source_path.suffix.lower() not in valid_extensions:
            return False

        # Try to open as Parquet file
        try:
            pq.ParquetFile(source_path)
            return True
        except Exception:
            return False

    def load_batch(
        self,
        source: str | Path,
        batch_size: int,
        name: str | None = None,
        **kwargs: Any,
    ) -> Iterator[Dataset]:
        """Load Parquet file in batches.

        Args:
            source: Path to Parquet file
            batch_size: Number of rows per batch
            name: Optional name prefix
            **kwargs: Additional options

        Yields:
            Dataset batches
        """
        source_path = Path(source)

        if not self.validate(source_path):
            raise DataValidationError(
                f"Invalid Parquet file: {source_path}", file_path=str(source_path)
            )

        dataset_name = name or source_path.stem
        target_column = kwargs.get("target_column")
        columns_to_read = kwargs.get("columns", self.columns)

        try:
            # Open Parquet file
            parquet_file = pq.ParquetFile(source_path)

            # Calculate batches based on row groups
            total_rows = parquet_file.metadata.num_rows
            rows_read = 0
            batch_index = 0

            # Read by row groups for efficiency
            for row_group_idx in range(parquet_file.num_row_groups):
                row_group = parquet_file.read_row_group(
                    row_group_idx, columns=columns_to_read
                )
                df_group = row_group.to_pandas()

                # Split row group into batches if needed
                group_size = len(df_group)

                for start_idx in range(0, group_size, batch_size):
                    end_idx = min(start_idx + batch_size, group_size)
                    batch_df = df_group.iloc[start_idx:end_idx]

                    if batch_df.empty:
                        continue

                    # Create dataset for this batch
                    batch_dataset = Dataset(
                        name=f"{dataset_name}_batch_{batch_index}",
                        data=batch_df,
                        target_column=target_column,
                        metadata={
                            "source": str(source_path),
                            "loader": "ParquetLoader",
                            "batch_index": batch_index,
                            "batch_size": len(batch_df),
                            "row_group": row_group_idx,
                            "total_rows": total_rows,
                            "rows_read": rows_read + len(batch_df),
                            "is_batch": True,
                        },
                    )

                    yield batch_dataset

                    batch_index += 1
                    rows_read += len(batch_df)

        except Exception as e:
            raise DataValidationError(
                f"Failed to load Parquet in batches: {e}", file_path=str(source_path)
            ) from e

    def estimate_size(self, source: str | Path) -> dict[str, Any]:
        """Estimate the size of the Parquet file.

        Args:
            source: Path to Parquet file

        Returns:
            Size information
        """
        source_path = Path(source)

        if not self.validate(source_path):
            raise DataValidationError(
                f"Invalid Parquet file: {source_path}", file_path=str(source_path)
            )

        try:
            # Open Parquet file for metadata
            parquet_file = pq.ParquetFile(source_path)
            metadata = parquet_file.metadata

            # Get basic info
            file_size_bytes = source_path.stat().st_size
            num_rows = metadata.num_rows
            num_columns = len(parquet_file.schema)

            # Get column types
            schema_info = {}
            numeric_columns = 0

            for field in parquet_file.schema:
                schema_info[field.name] = str(field.type)
                if (
                    "int" in str(field.type).lower()
                    or "float" in str(field.type).lower()
                    or "double" in str(field.type).lower()
                ):
                    numeric_columns += 1

            # Estimate memory usage
            # This is a rough estimate - actual usage depends on data
            # Parquet is compressed, so in-memory will be larger
            compression_ratio = 3  # Typical compression ratio
            estimated_memory_mb = (file_size_bytes * compression_ratio) / 1024 / 1024

            # Get row group info
            row_groups_info = []
            for i in range(parquet_file.num_row_groups):
                rg = metadata.row_group(i)
                row_groups_info.append(
                    {
                        "num_rows": rg.num_rows,
                        "total_byte_size": rg.total_byte_size,
                        "compressed_size": sum(
                            rg.column(j).total_compressed_size
                            for j in range(rg.num_columns)
                        ),
                    }
                )

            return {
                "file_size_mb": file_size_bytes / 1024 / 1024,
                "num_rows": num_rows,
                "num_columns": num_columns,
                "numeric_columns": numeric_columns,
                "estimated_memory_mb": estimated_memory_mb,
                "num_row_groups": metadata.num_row_groups,
                "schema": schema_info,
                "row_groups": row_groups_info,
                "created_by": str(metadata.created_by) if metadata.created_by else None,
                "format_version": str(metadata.format_version),
            }

        except Exception as e:
            return {
                "file_size_mb": source_path.stat().st_size / 1024 / 1024,
                "error": str(e),
            }
