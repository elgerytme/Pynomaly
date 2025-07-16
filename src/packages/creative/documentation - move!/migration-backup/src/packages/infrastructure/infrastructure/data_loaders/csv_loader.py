"""CSV data loader implementation."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd

from monorepo.domain.entities import Dataset
from monorepo.domain.exceptions import DataValidationError
from monorepo.shared.protocols import BatchDataLoaderProtocol


class CSVLoader(BatchDataLoaderProtocol):
    """Data loader for CSV files."""

    def __init__(
        self,
        delimiter: str = ",",
        encoding: str = "utf-8",
        parse_dates: bool = True,
        low_memory: bool = False,
    ):
        """Initialize CSV loader.

        Args:
            delimiter: Column delimiter
            encoding: File encoding
            parse_dates: Whether to parse date columns
            low_memory: Whether to use low memory mode
        """
        self.delimiter = delimiter
        self.encoding = encoding
        self.parse_dates = parse_dates
        self.low_memory = low_memory

    @property
    def supported_formats(self) -> list[str]:
        """Get supported file formats."""
        return ["csv", "tsv", "txt"]

    def load(
        self, source: str | Path, name: str | None = None, **kwargs: Any
    ) -> Dataset:
        """Load CSV file into a Dataset.

        Args:
            source: Path to CSV file
            name: Optional name for dataset
            **kwargs: Additional pandas read_csv arguments

        Returns:
            Loaded dataset
        """
        source_path = Path(source)

        if not self.validate(source_path):
            raise DataValidationError(
                f"Invalid CSV file: {source_path}", file_path=str(source_path)
            )

        # Prepare read options
        read_options = {
            "delimiter": kwargs.pop("delimiter", self.delimiter),
            "encoding": kwargs.pop("encoding", self.encoding),
            "parse_dates": kwargs.pop("parse_dates", self.parse_dates),
            "low_memory": kwargs.pop("low_memory", self.low_memory),
            **kwargs,
        }

        try:
            # Load data
            df = pd.read_csv(source_path, **read_options)

            # Handle empty dataframe
            if df.empty:
                raise DataValidationError(
                    "CSV file is empty", file_path=str(source_path)
                )

            # Create dataset
            dataset_name = name or source_path.stem

            # Check for target column
            target_column = kwargs.get("target_column")
            if target_column and target_column not in df.columns:
                raise DataValidationError(
                    f"Target column '{target_column}' not found in CSV",
                    file_path=str(source_path),
                    available_columns=list(df.columns),
                )

            dataset = Dataset(
                name=dataset_name,
                data=df,
                target_column=target_column,
                metadata={
                    "source": str(source_path),
                    "loader": "CSVLoader",
                    "delimiter": self.delimiter,
                    "encoding": self.encoding,
                    "file_size_mb": source_path.stat().st_size / 1024 / 1024,
                },
            )

            return dataset

        except pd.errors.EmptyDataError:
            raise DataValidationError(
                "CSV file is empty or corrupted", file_path=str(source_path)
            )
        except UnicodeDecodeError as e:
            raise DataValidationError(
                f"Encoding error: {e}. Try different encoding.",
                file_path=str(source_path),
                encoding=self.encoding,
            )
        except Exception as e:
            raise DataValidationError(
                f"Failed to load CSV: {e}", file_path=str(source_path)
            ) from e

    def validate(self, source: str | Path) -> bool:
        """Validate if source is a valid CSV file.

        Args:
            source: Path to validate

        Returns:
            True if valid CSV source
        """
        source_path = Path(source)

        # Check if file exists
        if not source_path.exists():
            return False

        # Check if it's a file
        if not source_path.is_file():
            return False

        # Check extension
        valid_extensions = {".csv", ".tsv", ".txt"}
        if source_path.suffix.lower() not in valid_extensions:
            return False

        # Try to read first few lines
        try:
            with open(source_path, encoding=self.encoding) as f:
                # Read first line to check if it's not empty
                first_line = f.readline()
                if not first_line.strip():
                    return False

                # Check if delimiter exists in first line
                if self.delimiter not in first_line:
                    # Maybe it's a single-column CSV
                    pass

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
        """Load CSV in batches.

        Args:
            source: Path to CSV file
            batch_size: Number of rows per batch
            name: Optional name prefix
            **kwargs: Additional options

        Yields:
            Dataset batches
        """
        source_path = Path(source)

        if not self.validate(source_path):
            raise DataValidationError(
                f"Invalid CSV file: {source_path}", file_path=str(source_path)
            )

        # Prepare read options
        read_options = {
            "delimiter": kwargs.pop("delimiter", self.delimiter),
            "encoding": kwargs.pop("encoding", self.encoding),
            "parse_dates": kwargs.pop("parse_dates", self.parse_dates),
            "low_memory": True,  # Always use low memory for batch loading
            "chunksize": batch_size,
            **kwargs,
        }

        dataset_name = name or source_path.stem
        target_column = kwargs.get("target_column")

        try:
            # Read in chunks
            chunk_iter = pd.read_csv(source_path, **read_options)

            for i, chunk in enumerate(chunk_iter):
                if chunk.empty:
                    continue

                # Create dataset for this batch
                batch_dataset = Dataset(
                    name=f"{dataset_name}_batch_{i}",
                    data=chunk,
                    target_column=target_column,
                    metadata={
                        "source": str(source_path),
                        "loader": "CSVLoader",
                        "batch_index": i,
                        "batch_size": batch_size,
                        "is_batch": True,
                    },
                )

                yield batch_dataset

        except Exception as e:
            raise DataValidationError(
                f"Failed to load CSV in batches: {e}", file_path=str(source_path)
            ) from e

    def estimate_size(self, source: str | Path) -> dict[str, Any]:
        """Estimate the size of the CSV file.

        Args:
            source: Path to CSV file

        Returns:
            Size information
        """
        source_path = Path(source)

        if not self.validate(source_path):
            raise DataValidationError(
                f"Invalid CSV file: {source_path}", file_path=str(source_path)
            )

        # Get file size
        file_size_bytes = source_path.stat().st_size

        # Estimate rows by sampling
        try:
            # Read first 1000 rows to estimate
            sample = pd.read_csv(
                source_path,
                nrows=1000,
                delimiter=self.delimiter,
                encoding=self.encoding,
            )

            if len(sample) == 0:
                return {
                    "file_size_mb": file_size_bytes / 1024 / 1024,
                    "estimated_rows": 0,
                    "columns": 0,
                    "memory_usage_mb": 0,
                }

            # Calculate average bytes per row
            sample_text = sample.to_csv(index=False)
            bytes_per_row = len(sample_text.encode()) / len(sample)

            # Estimate total rows
            estimated_rows = int(file_size_bytes / bytes_per_row)

            # Get column info
            n_columns = len(sample.columns)
            numeric_columns = len(sample.select_dtypes(include=["number"]).columns)

            # Estimate memory usage
            # Rough estimate: 8 bytes per numeric value, 50 bytes per string
            avg_string_cols = n_columns - numeric_columns
            estimated_memory = (
                (estimated_rows * (numeric_columns * 8 + avg_string_cols * 50))
                / 1024
                / 1024
            )

            return {
                "file_size_mb": file_size_bytes / 1024 / 1024,
                "estimated_rows": estimated_rows,
                "columns": n_columns,
                "numeric_columns": numeric_columns,
                "memory_usage_mb": estimated_memory,
                "sample_dtypes": sample.dtypes.to_dict(),
            }

        except Exception as e:
            # Fallback to basic info
            return {
                "file_size_mb": file_size_bytes / 1024 / 1024,
                "estimated_rows": "unknown",
                "error": str(e),
            }
