"""Enhanced Parquet data loader with advanced features."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from monorepo.domain.entities import Dataset
from monorepo.domain.exceptions import DataValidationError
from monorepo.shared.protocols import BatchDataLoaderProtocol


class EnhancedParquetLoader(BatchDataLoaderProtocol):
    """Enhanced Parquet data loader with advanced features and optimizations."""

    def __init__(
        self,
        engine: str = "pyarrow",
        use_memory_map: bool = True,
        columns: list[str] | None = None,
        filters: list[list[tuple]] | None = None,
        use_pandas_metadata: bool = True,
        validate_schema: bool = True,
    ):
        """Initialize enhanced Parquet loader.

        Args:
            engine: Parquet engine to use ('pyarrow' or 'fastparquet')
            use_memory_map: Whether to use memory mapping for better performance
            columns: Specific columns to load (None for all)
            filters: Row group filters for efficient loading
            use_pandas_metadata: Whether to use pandas metadata for type inference
            validate_schema: Whether to validate parquet schema
        """
        self.engine = engine
        self.use_memory_map = use_memory_map
        self.columns = columns
        self.filters = filters
        self.use_pandas_metadata = use_pandas_metadata
        self.validate_schema = validate_schema
        self.logger = logging.getLogger(__name__)

        # Validate engine availability
        if engine == "pyarrow":
            try:
                import pyarrow.parquet
            except ImportError:
                raise ImportError("PyArrow is required for pyarrow engine")
        elif engine == "fastparquet":
            try:
                import fastparquet
            except ImportError:
                raise ImportError("fastparquet is required for fastparquet engine")

    @property
    def supported_formats(self) -> list[str]:
        """Get supported file formats."""
        return ["parquet", "pq"]

    def load(
        self, source: str | Path, name: str | None = None, **kwargs: Any
    ) -> Dataset:
        """Load Parquet file into a Dataset.

        Args:
            source: Path to Parquet file or directory
            name: Optional name for dataset
            **kwargs: Additional pandas read_parquet arguments

        Returns:
            Loaded dataset
        """
        source_path = Path(source)

        if not self.validate(source_path):
            raise DataValidationError(
                f"Invalid Parquet source: {source_path}", file_path=str(source_path)
            )

        self.logger.info(f"Loading Parquet from {source_path}")

        # Prepare read options
        read_options = {
            "engine": kwargs.pop("engine", self.engine),
            "columns": kwargs.pop("columns", self.columns),
            "use_pandas_metadata": kwargs.pop(
                "use_pandas_metadata", self.use_pandas_metadata
            ),
            **kwargs,
        }

        # Add PyArrow specific options
        if self.engine == "pyarrow":
            read_options.update(
                {
                    "memory_map": kwargs.pop("memory_map", self.use_memory_map),
                    "filters": kwargs.pop("filters", self.filters),
                }
            )

        try:
            # Load data
            df = pd.read_parquet(source_path, **read_options)

            if df.empty:
                raise DataValidationError(
                    "Parquet file is empty", file_path=str(source_path)
                )

            # Get metadata from parquet file
            metadata = self._extract_metadata(source_path)

            # Create dataset
            dataset_name = name or source_path.stem

            # Check for target column
            target_column = kwargs.get("target_column")
            if target_column and target_column not in df.columns:
                raise DataValidationError(
                    f"Target column '{target_column}' not found",
                    file_path=str(source_path),
                    available_columns=list(df.columns),
                )

            dataset = Dataset(
                name=dataset_name,
                data=df,
                target_column=target_column,
                metadata={
                    "source": str(source_path),
                    "loader": "EnhancedParquetLoader",
                    "engine": self.engine,
                    "parquet_metadata": metadata,
                    "file_size_mb": self._get_file_size_mb(source_path),
                },
            )

            self.logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
            return dataset

        except Exception as e:
            raise DataValidationError(
                f"Failed to load Parquet file: {e}", file_path=str(source_path)
            ) from e

    def validate(self, source: str | Path) -> bool:
        """Validate if source is a valid Parquet file or directory.

        Args:
            source: Path to validate

        Returns:
            True if valid Parquet source
        """
        source_path = Path(source)

        # Check if exists
        if not source_path.exists():
            return False

        # Handle directory (partitioned parquet)
        if source_path.is_dir():
            # Check for parquet files in directory
            parquet_files = list(source_path.glob("*.parquet")) + list(
                source_path.glob("*.pq")
            )
            return len(parquet_files) > 0

        # Handle single file
        if not source_path.is_file():
            return False

        # Check extension
        valid_extensions = {".parquet", ".pq"}
        if source_path.suffix.lower() not in valid_extensions:
            return False

        # Try to read parquet metadata
        try:
            if self.engine == "pyarrow":
                pq.read_metadata(source_path)
            else:
                # For fastparquet, try to read schema
                import fastparquet

                fastparquet.ParquetFile(source_path)
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
        """Load Parquet in batches using row groups.

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

        self.logger.info(f"Loading Parquet in batches from {source_path}")

        dataset_name = name or source_path.stem
        target_column = kwargs.get("target_column")

        try:
            if self.engine == "pyarrow":
                # Use PyArrow for efficient batch reading
                parquet_file = pq.ParquetFile(source_path)

                # Read in batches of row groups
                row_group_batch_size = max(
                    1, batch_size // 1000
                )  # Adjust based on typical row group size

                for batch_idx in range(
                    0, parquet_file.num_row_groups, row_group_batch_size
                ):
                    end_idx = min(
                        batch_idx + row_group_batch_size, parquet_file.num_row_groups
                    )
                    row_groups = list(range(batch_idx, end_idx))

                    # Read batch
                    table = parquet_file.read_row_groups(
                        row_groups=row_groups,
                        columns=self.columns,
                        use_pandas_metadata=self.use_pandas_metadata,
                    )

                    batch_df = table.to_pandas()

                    if batch_df.empty:
                        continue

                    # Create dataset for this batch
                    batch_dataset = Dataset(
                        name=f"{dataset_name}_batch_{batch_idx}",
                        data=batch_df,
                        target_column=target_column,
                        metadata={
                            "source": str(source_path),
                            "loader": "EnhancedParquetLoader",
                            "batch_index": batch_idx,
                            "row_groups": row_groups,
                            "is_batch": True,
                        },
                    )

                    yield batch_dataset

            else:
                # Fallback to pandas chunking for other engines
                # This is less efficient but works with any engine
                full_df = pd.read_parquet(
                    source_path, engine=self.engine, columns=self.columns
                )

                for i in range(0, len(full_df), batch_size):
                    batch_df = full_df.iloc[i : i + batch_size].copy()

                    batch_dataset = Dataset(
                        name=f"{dataset_name}_batch_{i // batch_size}",
                        data=batch_df,
                        target_column=target_column,
                        metadata={
                            "source": str(source_path),
                            "loader": "EnhancedParquetLoader",
                            "batch_index": i // batch_size,
                            "batch_size": len(batch_df),
                            "is_batch": True,
                        },
                    )

                    yield batch_dataset

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
            file_size_mb = self._get_file_size_mb(source_path)

            if self.engine == "pyarrow":
                # Use PyArrow for detailed metadata
                parquet_file = pq.ParquetFile(source_path)
                metadata = parquet_file.metadata

                total_rows = metadata.num_rows
                num_columns = metadata.num_columns
                num_row_groups = metadata.num_row_groups

                # Estimate compressed vs uncompressed size
                compressed_size = metadata.serialized_size

                # Get column information
                schema = parquet_file.schema_arrow
                column_info = {}
                for i in range(len(schema)):
                    field = schema.field(i)
                    column_info[field.name] = str(field.type)

                return {
                    "file_size_mb": file_size_mb,
                    "total_rows": total_rows,
                    "num_columns": num_columns,
                    "num_row_groups": num_row_groups,
                    "compressed_size_mb": compressed_size / (1024 * 1024),
                    "column_types": column_info,
                    "avg_row_group_size": (
                        total_rows // num_row_groups if num_row_groups > 0 else 0
                    ),
                    "parquet_version": metadata.format_version,
                }

            else:
                # Fallback estimation for other engines
                return {
                    "file_size_mb": file_size_mb,
                    "estimated_rows": "unknown",
                    "columns": "unknown",
                    "engine": self.engine,
                }

        except Exception as e:
            return {
                "file_size_mb": self._get_file_size_mb(source_path),
                "error": str(e),
            }

    def get_schema_info(self, source: str | Path) -> dict[str, Any]:
        """Get detailed schema information from Parquet file.

        Args:
            source: Path to Parquet file

        Returns:
            Schema information
        """
        source_path = Path(source)

        if not self.validate(source_path):
            raise DataValidationError(
                f"Invalid Parquet file: {source_path}", file_path=str(source_path)
            )

        try:
            if self.engine == "pyarrow":
                parquet_file = pq.ParquetFile(source_path)
                schema = parquet_file.schema_arrow

                columns = []
                for i in range(len(schema)):
                    field = schema.field(i)
                    columns.append(
                        {
                            "name": field.name,
                            "type": str(field.type),
                            "nullable": field.nullable,
                            "metadata": field.metadata,
                        }
                    )

                return {
                    "columns": columns,
                    "num_columns": len(columns),
                    "schema_metadata": schema.metadata,
                    "pandas_metadata": (
                        schema.pandas_metadata
                        if hasattr(schema, "pandas_metadata")
                        else None
                    ),
                }

            else:
                # Basic schema info for other engines
                df_sample = pd.read_parquet(source_path, engine=self.engine, nrows=1)

                columns = []
                for col in df_sample.columns:
                    columns.append(
                        {
                            "name": col,
                            "type": str(df_sample[col].dtype),
                            "nullable": True,  # Default assumption
                        }
                    )

                return {
                    "columns": columns,
                    "num_columns": len(columns),
                }

        except Exception as e:
            raise DataValidationError(
                f"Failed to read schema: {e}", file_path=str(source_path)
            ) from e

    def _extract_metadata(self, source_path: Path) -> dict[str, Any]:
        """Extract metadata from Parquet file."""
        metadata = {}

        try:
            if self.engine == "pyarrow":
                parquet_file = pq.ParquetFile(source_path)
                pq_metadata = parquet_file.metadata

                metadata.update(
                    {
                        "num_rows": pq_metadata.num_rows,
                        "num_columns": pq_metadata.num_columns,
                        "num_row_groups": pq_metadata.num_row_groups,
                        "format_version": pq_metadata.format_version,
                        "created_by": pq_metadata.created_by,
                        "serialized_size": pq_metadata.serialized_size,
                    }
                )

                # Schema information
                schema = parquet_file.schema_arrow
                metadata["schema"] = {
                    "names": schema.names,
                    "types": [str(schema.field(i).type) for i in range(len(schema))],
                }

        except Exception as e:
            metadata["extraction_error"] = str(e)

        return metadata

    def _get_file_size_mb(self, source_path: Path) -> float:
        """Get file size in MB."""
        if source_path.is_file():
            return source_path.stat().st_size / (1024 * 1024)
        elif source_path.is_dir():
            # Sum all parquet files in directory
            total_size = 0
            for file_path in source_path.rglob("*.parquet"):
                total_size += file_path.stat().st_size
            for file_path in source_path.rglob("*.pq"):
                total_size += file_path.stat().st_size
            return total_size / (1024 * 1024)
        else:
            return 0.0
