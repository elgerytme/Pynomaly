"""JSON data loader implementation."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd

from monorepo.domain.entities import Dataset
from monorepo.domain.exceptions import DataValidationError
from monorepo.shared.protocols import BatchDataLoaderProtocol


class JSONLoader(BatchDataLoaderProtocol):
    """Data loader for JSON files."""

    def __init__(
        self,
        encoding: str = "utf-8",
        lines: bool = False,
        normalize_nested: bool = True,
    ):
        """Initialize JSON loader.

        Args:
            encoding: File encoding
            lines: Whether to treat as JSON Lines format
            normalize_nested: Whether to normalize nested objects
        """
        self.encoding = encoding
        self.lines = lines
        self.normalize_nested = normalize_nested

    @property
    def supported_formats(self) -> list[str]:
        """Get supported file formats."""
        return ["json", "jsonl"]

    def load(
        self, source: str | Path, name: str | None = None, **kwargs: Any
    ) -> Dataset:
        """Load JSON file into a Dataset.

        Args:
            source: Path to JSON file
            name: Optional name for dataset
            **kwargs: Additional options

        Returns:
            Loaded dataset
        """
        source_path = Path(source)

        if not self.validate(source_path):
            raise DataValidationError(
                f"Invalid JSON file: {source_path}", file_path=str(source_path)
            )

        try:
            # Determine if it's JSON Lines format
            is_lines = kwargs.get("lines", self.lines)
            if source_path.suffix.lower() == ".jsonl":
                is_lines = True

            # Load data
            if is_lines:
                df = pd.read_json(source_path, lines=True, encoding=self.encoding)
            else:
                df = pd.read_json(source_path, encoding=self.encoding)

            # Normalize nested objects if requested
            if self.normalize_nested and not df.empty:
                # Find columns with nested objects/arrays
                for col in df.columns:
                    if df[col].dtype == "object":
                        sample_val = (
                            df[col].dropna().iloc[0]
                            if not df[col].dropna().empty
                            else None
                        )
                        if isinstance(sample_val, dict | list):
                            # Normalize nested data
                            try:
                                normalized = pd.json_normalize(df[col])
                                normalized.columns = [
                                    f"{col}_{subcol}" for subcol in normalized.columns
                                ]
                                df = df.drop(columns=[col])
                                df = pd.concat([df, normalized], axis=1)
                            except Exception:
                                # Keep original if normalization fails
                                pass

            if df.empty:
                raise DataValidationError(
                    "JSON file is empty", file_path=str(source_path)
                )

            # Create dataset
            dataset_name = name or source_path.stem
            target_column = kwargs.get("target_column")

            if target_column and target_column not in df.columns:
                raise DataValidationError(
                    f"Target column '{target_column}' not found in JSON",
                    file_path=str(source_path),
                    available_columns=list(df.columns),
                )

            dataset = Dataset(
                name=dataset_name,
                data=df,
                target_column=target_column,
                metadata={
                    "source": str(source_path),
                    "loader": "JSONLoader",
                    "encoding": self.encoding,
                    "lines_format": is_lines,
                    "normalized": self.normalize_nested,
                    "file_size_mb": source_path.stat().st_size / 1024 / 1024,
                },
            )

            return dataset

        except json.JSONDecodeError as e:
            raise DataValidationError(
                f"Invalid JSON format: {e}", file_path=str(source_path)
            )
        except UnicodeDecodeError as e:
            raise DataValidationError(
                f"Encoding error: {e}. Try different encoding.",
                file_path=str(source_path),
                encoding=self.encoding,
            )
        except Exception as e:
            raise DataValidationError(
                f"Failed to load JSON: {e}", file_path=str(source_path)
            ) from e

    def validate(self, source: str | Path) -> bool:
        """Validate if source is a valid JSON file.

        Args:
            source: Path to validate

        Returns:
            True if valid JSON source
        """
        source_path = Path(source)

        # Check if file exists
        if not source_path.exists() or not source_path.is_file():
            return False

        # Check extension
        valid_extensions = {".json", ".jsonl"}
        if source_path.suffix.lower() not in valid_extensions:
            return False

        # Try to parse JSON
        try:
            with open(source_path, encoding=self.encoding) as f:
                if source_path.suffix.lower() == ".jsonl":
                    # For JSONL, just check first line
                    first_line = f.readline().strip()
                    if first_line:
                        json.loads(first_line)
                else:
                    # For JSON, parse entire file (but limit size)
                    content = f.read(10000)  # First 10KB
                    if content.strip():
                        json.loads(content)

            return True

        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            return False

    def load_batch(
        self,
        source: str | Path,
        batch_size: int,
        name: str | None = None,
        **kwargs: Any,
    ) -> Iterator[Dataset]:
        """Load JSON in batches (mainly for JSONL format).

        Args:
            source: Path to JSON file
            batch_size: Number of records per batch
            name: Optional name prefix
            **kwargs: Additional options

        Yields:
            Dataset batches
        """
        source_path = Path(source)

        if not self.validate(source_path):
            raise DataValidationError(
                f"Invalid JSON file: {source_path}", file_path=str(source_path)
            )

        dataset_name = name or source_path.stem
        target_column = kwargs.get("target_column")
        is_lines = (
            kwargs.get("lines", self.lines) or source_path.suffix.lower() == ".jsonl"
        )

        if not is_lines:
            # For regular JSON, load all and split into batches
            full_dataset = self.load(source, name, **kwargs)
            df = full_dataset.data

            for i in range(0, len(df), batch_size):
                batch_df = df.iloc[i : i + batch_size]

                batch_dataset = Dataset(
                    name=f"{dataset_name}_batch_{i // batch_size}",
                    data=batch_df,
                    target_column=target_column,
                    metadata={
                        "source": str(source_path),
                        "loader": "JSONLoader",
                        "batch_index": i // batch_size,
                        "batch_size": batch_size,
                        "is_batch": True,
                    },
                )

                yield batch_dataset
        else:
            # For JSONL, read line by line
            try:
                with open(source_path, encoding=self.encoding) as f:
                    batch_records = []
                    batch_index = 0

                    for line in f:
                        line = line.strip()
                        if not line:
                            continue

                        try:
                            record = json.loads(line)
                            batch_records.append(record)

                            if len(batch_records) >= batch_size:
                                # Convert to DataFrame
                                batch_df = pd.DataFrame(batch_records)

                                # Normalize if requested
                                if self.normalize_nested:
                                    # Simple normalization for batch
                                    for col in batch_df.columns:
                                        if batch_df[col].dtype == "object":
                                            sample_val = (
                                                batch_df[col].dropna().iloc[0]
                                                if not batch_df[col].dropna().empty
                                                else None
                                            )
                                            if isinstance(sample_val, dict):
                                                try:
                                                    normalized = pd.json_normalize(
                                                        batch_df[col]
                                                    )
                                                    normalized.columns = [
                                                        f"{col}_{subcol}"
                                                        for subcol in normalized.columns
                                                    ]
                                                    batch_df = batch_df.drop(
                                                        columns=[col]
                                                    )
                                                    batch_df = pd.concat(
                                                        [batch_df, normalized], axis=1
                                                    )
                                                except Exception:
                                                    pass

                                batch_dataset = Dataset(
                                    name=f"{dataset_name}_batch_{batch_index}",
                                    data=batch_df,
                                    target_column=target_column,
                                    metadata={
                                        "source": str(source_path),
                                        "loader": "JSONLoader",
                                        "batch_index": batch_index,
                                        "batch_size": batch_size,
                                        "is_batch": True,
                                        "lines_format": True,
                                    },
                                )

                                yield batch_dataset

                                batch_records = []
                                batch_index += 1

                        except json.JSONDecodeError:
                            continue  # Skip invalid lines

                    # Yield remaining records
                    if batch_records:
                        batch_df = pd.DataFrame(batch_records)

                        batch_dataset = Dataset(
                            name=f"{dataset_name}_batch_{batch_index}",
                            data=batch_df,
                            target_column=target_column,
                            metadata={
                                "source": str(source_path),
                                "loader": "JSONLoader",
                                "batch_index": batch_index,
                                "batch_size": len(batch_records),
                                "is_batch": True,
                                "lines_format": True,
                            },
                        )

                        yield batch_dataset

            except Exception as e:
                raise DataValidationError(
                    f"Failed to load JSON in batches: {e}", file_path=str(source_path)
                ) from e

    def estimate_size(self, source: str | Path) -> dict[str, Any]:
        """Estimate the size of the JSON file.

        Args:
            source: Path to JSON file

        Returns:
            Size information
        """
        source_path = Path(source)

        if not self.validate(source_path):
            raise DataValidationError(
                f"Invalid JSON file: {source_path}", file_path=str(source_path)
            )

        file_size_bytes = source_path.stat().st_size

        try:
            # Sample first part of file to estimate structure
            with open(source_path, encoding=self.encoding) as f:
                is_lines = source_path.suffix.lower() == ".jsonl"

                if is_lines:
                    # Count lines for JSONL
                    sample_lines = []
                    for i, line in enumerate(f):
                        if i >= 100:  # Sample first 100 lines
                            break
                        line = line.strip()
                        if line:
                            try:
                                sample_lines.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue

                    if sample_lines:
                        # Estimate total lines
                        avg_line_size = file_size_bytes / 100  # Rough estimate
                        estimated_rows = max(
                            len(sample_lines), int(file_size_bytes / avg_line_size)
                        )

                        # Analyze structure
                        sample_df = pd.DataFrame(sample_lines)
                        n_columns = len(sample_df.columns)

                        return {
                            "file_size_mb": file_size_bytes / 1024 / 1024,
                            "estimated_rows": estimated_rows,
                            "columns": n_columns,
                            "format": "jsonl",
                            "sample_columns": list(sample_df.columns),
                        }
                else:
                    # Regular JSON - load and analyze
                    sample_size = min(file_size_bytes, 100000)  # First 100KB
                    sample_content = f.read(sample_size)

                    try:
                        sample_data = json.loads(sample_content)

                        if isinstance(sample_data, list):
                            estimated_rows = len(sample_data)
                            if sample_data:
                                sample_df = pd.DataFrame(
                                    sample_data[:100]
                                )  # First 100 records
                                n_columns = len(sample_df.columns)
                            else:
                                n_columns = 0
                        elif isinstance(sample_data, dict):
                            # Single record or nested structure
                            sample_df = pd.json_normalize([sample_data])
                            estimated_rows = 1
                            n_columns = len(sample_df.columns)
                        else:
                            estimated_rows = 1
                            n_columns = 1

                        return {
                            "file_size_mb": file_size_bytes / 1024 / 1024,
                            "estimated_rows": estimated_rows,
                            "columns": n_columns,
                            "format": "json",
                            "data_type": type(sample_data).__name__,
                        }

                    except json.JSONDecodeError:
                        return {
                            "file_size_mb": file_size_bytes / 1024 / 1024,
                            "estimated_rows": "unknown",
                            "error": "Invalid JSON format",
                        }

        except Exception as e:
            return {
                "file_size_mb": file_size_bytes / 1024 / 1024,
                "estimated_rows": "unknown",
                "error": str(e),
            }

        return {
            "file_size_mb": file_size_bytes / 1024 / 1024,
            "estimated_rows": 0,
            "columns": 0,
        }
