"""Data loading service for autonomous processing."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from monorepo.domain.entities import Dataset
from monorepo.domain.exceptions import DataValidationError
from monorepo.shared.protocols import DataLoaderProtocol


class DataLoaderService:
    """Service responsible for automatic data loading and format processing."""

    def __init__(self, data_loaders: dict[str, DataLoaderProtocol]):
        """Initialize data loader service.

        Args:
            data_loaders: Dictionary of data loaders by format
        """
        self.data_loaders = data_loaders
        self.logger = logging.getLogger(__name__)

    async def load_data(
        self,
        data_source: str | Path | pd.DataFrame,
        name: str = "autonomous_data",
        verbose: bool = False,
    ) -> DataCollection:
        """Automatically detect and load data source.

        Args:
            data_source: Path to data file, connection string, or DataFrame
            name: Name for the data_collection
            verbose: Enable verbose logging

        Returns:
            Loaded data_collection

        Raises:
            DataValidationError: If data format is unsupported
        """
        if isinstance(data_source, pd.DataFrame):
            # Direct DataFrame
            return DataCollection(
                name=name,
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

        if verbose:
            self.logger.info(f"Loading {format_type} data from {source_path}")

        return loader.load(source_path, name=name, **load_options)

    def _detect_data_format(self, source_path: Path) -> str:
        """Detect data format from file extension and content.

        Args:
            source_path: Path to the data source

        Returns:
            Detected format type
        """
        extension = source_path.suffix.lower()

        # Extension-based processing
        format_map = {
            ".csv": "csv",
            ".json": "json",
            ".jsonl": "jsonl",
            ".parquet": "parquet",
            ".xlsx": "excel",
            ".xls": "excel",
            ".feather": "feather",
            ".pickle": "pickle",
            ".pkl": "pickle",
            ".h5": "hdf5",
            ".hdf5": "hdf5",
            ".sql": "sql",
            ".db": "sqlite",
            ".sqlite": "sqlite",
            ".tsv": "tsv",
            ".txt": "txt",
            ".log": "log",
            ".avro": "avro",
            ".orc": "orc",
        }

        if extension in format_map:
            return format_map[extension]

        # Content-based processing for ambiguous extensions
        if extension in [".txt", ".log"]:
            return self._detect_text_format(source_path)

        # Default to CSV for unknown extensions
        self.logger.warning(f"Unknown extension {extension}, defaulting to CSV")
        return "csv"

    def _detect_text_format(self, source_path: Path) -> str:
        """Detect text format by examining file content.

        Args:
            source_path: Path to the text file

        Returns:
            Detected text format
        """
        try:
            with source_path.open("r", encoding="utf-8") as f:
                # Read first few lines to detect format
                lines = [f.readline().strip() for _ in range(5)]
                content = "\n".join(lines)

                # Check for JSON
                if content.startswith("{") or content.startswith("["):
                    return "json"

                # Check for CSV (comma-separated)
                if "," in content and len(content.split(",")) > 1:
                    return "csv"

                # Check for TSV (tab-separated)
                if "\t" in content and len(content.split("\t")) > 1:
                    return "tsv"

                # Check for log format
                if any(
                    keyword in content.lower()
                    for keyword in ["log", "error", "warning", "info", "debug"]
                ):
                    return "log"

                # Default to plain text
                return "txt"

        except Exception as e:
            self.logger.warning(f"Failed to detect text format: {e}")
            return "txt"

    def _auto_configure_loader(self, source_path: Path, format_type: str) -> dict[str, Any]:
        """Auto-configure loader options based on format and content.

        Args:
            source_path: Path to the data source
            format_type: Detected format type

        Returns:
            Configuration options for the loader
        """
        options = {}

        if format_type == "csv":
            options.update(self._configure_csv_loader(source_path))
        elif format_type == "json":
            options.update(self._configure_json_loader(source_path))
        elif format_type == "excel":
            options.update(self._configure_excel_loader(source_path))
        elif format_type == "tsv":
            options.update({"delimiter": "\t"})
        elif format_type == "log":
            options.update(self._configure_log_loader(source_path))

        return options

    def _configure_csv_loader(self, source_path: Path) -> dict[str, Any]:
        """Configure CSV loader options.

        Args:
            source_path: Path to the CSV file

        Returns:
            CSV loader configuration
        """
        options = {
            "delimiter": ",",
            "header": 0,
            "index_col": None,
            "parse_dates": True,
            "infer_datetime_format": True,
        }

        try:
            # Sample first few lines to detect delimiter and header
            with source_path.open("r", encoding="utf-8") as f:
                first_lines = [f.readline().strip() for _ in range(5)]

            if first_lines:
                # Detect delimiter
                for delimiter in [",", ";", "\t", "|"]:
                    if delimiter in first_lines[0]:
                        options["delimiter"] = delimiter
                        break

                # Detect header
                if first_lines[0] and not first_lines[0][0].isdigit():
                    options["header"] = 0
                else:
                    options["header"] = None

        except Exception as e:
            self.logger.warning(f"Failed to configure CSV loader: {e}")

        return options

    def _configure_json_loader(self, source_path: Path) -> dict[str, Any]:
        """Configure JSON loader options.

        Args:
            source_path: Path to the JSON file

        Returns:
            JSON loader configuration
        """
        options = {
            "orient": "records",
            "lines": False,
        }

        try:
            # Check if it's JSON Lines format
            with source_path.open("r", encoding="utf-8") as f:
                first_line = f.readline().strip()
                if first_line.startswith("{") and not first_line.endswith("}"):
                    options["lines"] = True

        except Exception as e:
            self.logger.warning(f"Failed to configure JSON loader: {e}")

        return options

    def _configure_excel_loader(self, source_path: Path) -> dict[str, Any]:
        """Configure Excel loader options.

        Args:
            source_path: Path to the Excel file

        Returns:
            Excel loader configuration
        """
        options = {
            "sheet_name": 0,
            "header": 0,
            "index_col": None,
            "parse_dates": True,
        }

        try:
            # Try to detect sheet names and structure
            import openpyxl

            workbook = openpyxl.load_workbook(source_path, read_only=True)
            sheet_names = workbook.sheetnames

            if sheet_names:
                # Use first sheet by default
                options["sheet_name"] = sheet_names[0]

            workbook.close()

        except Exception as e:
            self.logger.warning(f"Failed to configure Excel loader: {e}")

        return options

    def _configure_log_loader(self, source_path: Path) -> dict[str, Any]:
        """Configure log loader options.

        Args:
            source_path: Path to the log file

        Returns:
            Log loader configuration
        """
        options = {
            "delimiter": " ",
            "header": None,
            "parse_dates": True,
            "date_parser": None,
        }

        try:
            # Sample log format to detect structure
            with source_path.open("r", encoding="utf-8") as f:
                sample_lines = [f.readline().strip() for _ in range(10)]

            # Try to detect common log patterns
            if sample_lines:
                # Check for timestamp patterns
                import re

                timestamp_patterns = [
                    r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}",  # YYYY-MM-DD HH:MM:SS
                    r"\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}",  # MM/DD/YYYY HH:MM:SS
                    r"\w{3} \d{2} \d{2}:\d{2}:\d{2}",        # Mon DD HH:MM:SS
                ]

                for pattern in timestamp_patterns:
                    if re.search(pattern, sample_lines[0]):
                        options["parse_dates"] = [0]
                        break

        except Exception as e:
            self.logger.warning(f"Failed to configure log loader: {e}")

        return options

    def get_supported_formats(self) -> list[str]:
        """Get list of supported data formats.

        Returns:
            List of supported format names
        """
        return list(self.data_loaders.keys())

    def is_format_supported(self, format_type: str) -> bool:
        """Check if a format is supported.

        Args:
            format_type: Format type to check

        Returns:
            True if format is supported, False otherwise
        """
        return format_type in self.data_loaders
