"""File-based data loaders for various formats."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from pynomaly.domain.entities.dataset import Dataset

logger = logging.getLogger(__name__)


class DataLoader(Protocol):
    """Protocol for data loaders."""

    def load(self, source: str | Path, **kwargs: Any) -> Dataset:
        """Load data from source."""
        ...

    def supports_format(self, file_path: str | Path) -> bool:
        """Check if loader supports the file format."""
        ...


class CSVLoader:
    """CSV file data loader."""

    def __init__(self) -> None:
        """Initialize CSV loader."""
        self.supported_extensions = {".csv", ".tsv", ".txt"}

    def load(self, source: str | Path, **kwargs: Any) -> Dataset:
        """Load CSV data."""
        file_path = Path(source)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not self.supports_format(file_path):
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        # Set default CSV parameters
        csv_params = {
            "encoding": "utf-8",
            "sep": "," if file_path.suffix == ".csv" else "\t",
            "index_col": None,
            "parse_dates": False,
        }
        csv_params.update(kwargs)

        try:
            logger.info(f"Loading CSV file: {file_path}")
            data = pd.read_csv(file_path, **csv_params)

            # Validate data
            if data.empty:
                raise ValueError("Loaded dataset is empty")

            # Create dataset
            dataset = Dataset(
                name=file_path.stem,
                data=data,
                metadata={
                    "source_file": str(file_path),
                    "file_size": file_path.stat().st_size,
                    "loader": "CSVLoader",
                    "load_params": csv_params,
                },
            )

            logger.info(
                f"CSV loaded successfully: {data.shape[0]} rows, {data.shape[1]} columns"
            )
            return dataset

        except Exception as e:
            logger.error(f"Failed to load CSV file {file_path}: {e}")
            raise RuntimeError(f"Failed to load CSV: {e}") from e

    def supports_format(self, file_path: str | Path) -> bool:
        """Check if file format is supported."""
        return Path(file_path).suffix.lower() in self.supported_extensions


class JSONLoader:
    """JSON file data loader."""

    def __init__(self) -> None:
        """Initialize JSON loader."""
        self.supported_extensions = {".json", ".jsonl", ".ndjson"}

    def load(self, source: str | Path, **kwargs: Any) -> Dataset:
        """Load JSON data."""
        file_path = Path(source)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not self.supports_format(file_path):
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        try:
            logger.info(f"Loading JSON file: {file_path}")

            # Handle different JSON formats
            if file_path.suffix.lower() in {".jsonl", ".ndjson"}:
                # Line-delimited JSON
                data = pd.read_json(file_path, lines=True, **kwargs)
            else:
                # Regular JSON
                with open(file_path, "r", encoding="utf-8") as f:
                    json_data = json.load(f)

                # Convert to DataFrame
                if isinstance(json_data, list):
                    data = pd.DataFrame(json_data)
                elif isinstance(json_data, dict):
                    # Handle different dict structures
                    if "data" in json_data:
                        data = pd.DataFrame(json_data["data"])
                    else:
                        data = pd.DataFrame([json_data])
                else:
                    raise ValueError(
                        "JSON structure not supported for conversion to DataFrame"
                    )

            # Validate data
            if data.empty:
                raise ValueError("Loaded dataset is empty")

            # Create dataset
            dataset = Dataset(
                name=file_path.stem,
                data=data,
                metadata={
                    "source_file": str(file_path),
                    "file_size": file_path.stat().st_size,
                    "loader": "JSONLoader",
                    "json_format": (
                        "lines"
                        if file_path.suffix.lower() in {".jsonl", ".ndjson"}
                        else "standard"
                    ),
                },
            )

            logger.info(
                f"JSON loaded successfully: {data.shape[0]} rows, {data.shape[1]} columns"
            )
            return dataset

        except Exception as e:
            logger.error(f"Failed to load JSON file {file_path}: {e}")
            raise RuntimeError(f"Failed to load JSON: {e}") from e

    def supports_format(self, file_path: str | Path) -> bool:
        """Check if file format is supported."""
        return Path(file_path).suffix.lower() in self.supported_extensions


class ParquetLoader:
    """Parquet file data loader."""

    def __init__(self) -> None:
        """Initialize Parquet loader."""
        self.supported_extensions = {".parquet", ".pq"}

    def load(self, source: str | Path, **kwargs: Any) -> Dataset:
        """Load Parquet data."""
        file_path = Path(source)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not self.supports_format(file_path):
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        try:
            # Check if pyarrow is available
            try:
                import pyarrow.parquet as pq

                engine = "pyarrow"
            except ImportError:
                try:
                    import fastparquet

                    engine = "fastparquet"
                except ImportError:
                    raise ImportError(
                        "Parquet support requires either pyarrow or fastparquet. "
                        "Install with: pip install pyarrow"
                    )

            logger.info(f"Loading Parquet file: {file_path} using {engine}")

            # Set default parameters
            parquet_params = {"engine": engine}
            parquet_params.update(kwargs)

            data = pd.read_parquet(file_path, **parquet_params)

            # Validate data
            if data.empty:
                raise ValueError("Loaded dataset is empty")

            # Create dataset
            dataset = Dataset(
                name=file_path.stem,
                data=data,
                metadata={
                    "source_file": str(file_path),
                    "file_size": file_path.stat().st_size,
                    "loader": "ParquetLoader",
                    "engine": engine,
                },
            )

            logger.info(
                f"Parquet loaded successfully: {data.shape[0]} rows, {data.shape[1]} columns"
            )
            return dataset

        except Exception as e:
            logger.error(f"Failed to load Parquet file {file_path}: {e}")
            raise RuntimeError(f"Failed to load Parquet: {e}") from e

    def supports_format(self, file_path: str | Path) -> bool:
        """Check if file format is supported."""
        return Path(file_path).suffix.lower() in self.supported_extensions


class ExcelLoader:
    """Excel file data loader."""

    def __init__(self) -> None:
        """Initialize Excel loader."""
        self.supported_extensions = {".xlsx", ".xls", ".xlsm"}

    def load(self, source: str | Path, **kwargs: Any) -> Dataset:
        """Load Excel data."""
        file_path = Path(source)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not self.supports_format(file_path):
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        try:
            logger.info(f"Loading Excel file: {file_path}")

            # Set default parameters
            excel_params = {
                "sheet_name": 0,  # First sheet by default
                "header": 0,
                "index_col": None,
            }
            excel_params.update(kwargs)

            data = pd.read_excel(file_path, **excel_params)

            # Validate data
            if data.empty:
                raise ValueError("Loaded dataset is empty")

            # Create dataset
            dataset = Dataset(
                name=file_path.stem,
                data=data,
                metadata={
                    "source_file": str(file_path),
                    "file_size": file_path.stat().st_size,
                    "loader": "ExcelLoader",
                    "sheet_name": excel_params["sheet_name"],
                },
            )

            logger.info(
                f"Excel loaded successfully: {data.shape[0]} rows, {data.shape[1]} columns"
            )
            return dataset

        except Exception as e:
            logger.error(f"Failed to load Excel file {file_path}: {e}")
            raise RuntimeError(f"Failed to load Excel: {e}") from e

    def supports_format(self, file_path: str | Path) -> bool:
        """Check if file format is supported."""
        return Path(file_path).suffix.lower() in self.supported_extensions


class AutoDataLoader:
    """Automatic data loader that detects file format."""

    def __init__(self) -> None:
        """Initialize auto loader with all supported loaders."""
        self._loaders = [
            CSVLoader(),
            JSONLoader(),
            ParquetLoader(),
            ExcelLoader(),
        ]

    def load(self, source: str | Path, **kwargs: Any) -> Dataset:
        """Load data by automatically detecting format."""
        file_path = Path(source)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Find appropriate loader
        for loader in self._loaders:
            if loader.supports_format(file_path):
                logger.info(f"Using {loader.__class__.__name__} for {file_path}")
                return loader.load(source, **kwargs)

        # No loader found
        supported_formats = []
        for loader in self._loaders:
            supported_formats.extend(loader.supported_extensions)

        raise ValueError(
            f"Unsupported file format: {file_path.suffix}. "
            f"Supported formats: {sorted(set(supported_formats))}"
        )

    def get_supported_formats(self) -> list[str]:
        """Get all supported file formats."""
        formats = []
        for loader in self._loaders:
            formats.extend(loader.supported_extensions)
        return sorted(set(formats))


class DataLoaderService:
    """Service for loading data from various sources."""

    def __init__(self) -> None:
        """Initialize data loader service."""
        self._auto_loader = AutoDataLoader()
        self._loaders = {
            "csv": CSVLoader(),
            "json": JSONLoader(),
            "parquet": ParquetLoader(),
            "excel": ExcelLoader(),
            "auto": self._auto_loader,
        }

    def load_file(
        self, file_path: str | Path, loader_type: str = "auto", **kwargs: Any
    ) -> Dataset:
        """Load data from file using specified or auto-detected loader."""
        if loader_type not in self._loaders:
            raise ValueError(f"Unknown loader type: {loader_type}")

        loader = self._loaders[loader_type]

        try:
            return loader.load(file_path, **kwargs)
        except Exception as e:
            logger.error(
                f"Failed to load file {file_path} with {loader_type} loader: {e}"
            )
            raise

    def load_csv(self, file_path: str | Path, **kwargs: Any) -> Dataset:
        """Load CSV file."""
        return self._loaders["csv"].load(file_path, **kwargs)

    def load_json(self, file_path: str | Path, **kwargs: Any) -> Dataset:
        """Load JSON file."""
        return self._loaders["json"].load(file_path, **kwargs)

    def load_parquet(self, file_path: str | Path, **kwargs: Any) -> Dataset:
        """Load Parquet file."""
        return self._loaders["parquet"].load(file_path, **kwargs)

    def load_excel(self, file_path: str | Path, **kwargs: Any) -> Dataset:
        """Load Excel file."""
        return self._loaders["excel"].load(file_path, **kwargs)

    def get_supported_formats(self) -> dict[str, list[str]]:
        """Get supported formats for each loader."""
        return {
            loader_name: (
                loader.get_supported_formats()
                if hasattr(loader, "get_supported_formats")
                else getattr(loader, "supported_extensions", [])
            )
            for loader_name, loader in self._loaders.items()
            if loader_name != "auto"
        }

    def validate_file(self, file_path: str | Path) -> dict[str, Any]:
        """Validate file and return metadata."""
        file_path = Path(file_path)

        if not file_path.exists():
            return {"valid": False, "error": "File not found"}

        if not file_path.is_file():
            return {"valid": False, "error": "Path is not a file"}

        # Check if format is supported
        supported = False
        detected_loader = None

        for loader_name, loader in self._loaders.items():
            if loader_name != "auto" and hasattr(loader, "supports_format"):
                if loader.supports_format(file_path):
                    supported = True
                    detected_loader = loader_name
                    break

        return {
            "valid": supported,
            "file_path": str(file_path),
            "file_size": file_path.stat().st_size,
            "file_extension": file_path.suffix,
            "detected_loader": detected_loader,
            "error": None if supported else f"Unsupported format: {file_path.suffix}",
        }
