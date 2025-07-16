"""Autonomous data loading and format detection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from pynomaly_detection.application.services.autonomous_detection_config import AutonomousConfig
from pynomaly_detection.domain.entities import Dataset
from pynomaly_detection.domain.exceptions import DataValidationError
from pynomaly_detection.shared.protocols import DataLoaderProtocol

logger = logging.getLogger(__name__)


class AutonomousDataLoader:
    """Service for autonomous data loading with format detection."""

    def __init__(self, data_loaders: dict[str, DataLoaderProtocol]):
        """Initialize data loader.

        Args:
            data_loaders: Dictionary of data loaders by format type
        """
        self.data_loaders = data_loaders

    async def auto_load_data(
        self, data_source: str | Path | pd.DataFrame, config: AutonomousConfig
    ) -> Dataset:
        """Automatically detect and load data source.

        Args:
            data_source: Path to data file, connection string, or DataFrame
            config: Configuration options

        Returns:
            Loaded dataset
        """
        if isinstance(data_source, pd.DataFrame):
            # Direct DataFrame
            return Dataset(
                name="autonomous_data",
                data=data_source,
                metadata={"source": "dataframe", "loader": "direct"},
            )

        source_path = Path(data_source)

        # Detect data format
        format_type = self._detect_data_format(source_path)

        if format_type not in self.data_loaders:
            raise DataValidationError(f"Unsupported data format: {format_type}")

        loader = self.data_loaders[format_type]

        # Auto-configure loader options
        load_options = self._auto_configure_loader(source_path, format_type)

        if config.verbose:
            logger.info(f"Loading {format_type} data from {source_path}")

        return loader.load(source_path, name="autonomous_data", **load_options)

    def _detect_data_format(self, source_path: Path) -> str:
        """Detect data format from file extension and content.

        Args:
            source_path: Path to the data file

        Returns:
            Detected format type
        """
        extension = source_path.suffix.lower()

        # Extension-based detection
        format_map = {
            ".csv": "csv",
            ".tsv": "csv",
            ".txt": "csv",
            ".parquet": "parquet",
            ".pq": "parquet",
            ".arrow": "arrow",
            ".xlsx": "excel",
            ".xls": "excel",
            ".json": "json",
            ".jsonl": "json",
        }

        if extension in format_map:
            return format_map[extension]

        # Content-based detection for ambiguous files
        try:
            with open(source_path, encoding="utf-8") as f:
                first_line = f.readline()

                # Check for CSV-like delimiters
                if any(delimiter in first_line for delimiter in [",", "\t", ";", "|"]):
                    return "csv"

                # Check for JSON
                if first_line.strip().startswith("{") or first_line.strip().startswith(
                    "["
                ):
                    return "json"

        except (UnicodeDecodeError, OSError):
            pass

        # Default to CSV for unknown text files
        return "csv"

    def _auto_configure_loader(
        self, source_path: Path, format_type: str
    ) -> dict[str, Any]:
        """Auto-configure data loader options.

        Args:
            source_path: Path to the data file
            format_type: Detected format type

        Returns:
            Dictionary of loader options
        """
        options = {}

        if format_type == "csv":
            # Auto-detect delimiter
            try:
                with open(source_path, encoding="utf-8") as f:
                    first_line = f.readline()

                    # Count delimiter candidates
                    delimiters = [",", "\t", ";", "|"]
                    delimiter_counts = {d: first_line.count(d) for d in delimiters}

                    # Choose most common delimiter
                    best_delimiter = max(delimiter_counts, key=delimiter_counts.get)
                    if delimiter_counts[best_delimiter] > 0:
                        options["delimiter"] = best_delimiter

            except (OSError, UnicodeDecodeError):
                pass

            # Auto-detect encoding
            try:
                import chardet

                with open(source_path, "rb") as f:
                    raw_data = f.read(10000)
                    encoding_result = chardet.detect(raw_data)
                    if encoding_result["confidence"] > 0.8:
                        options["encoding"] = encoding_result["encoding"]
            except (ImportError, OSError):
                pass

        return options
