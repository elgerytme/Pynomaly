"""Data loader factory for creating appropriate loaders based on data source."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pynomaly.domain.entities import Dataset
from pynomaly.domain.exceptions import DataValidationError
from pynomaly.shared.protocols import DataLoaderProtocol

# Import all available loaders
from .csv_loader import CSVLoader
from .excel_loader import ExcelLoader
from .json_loader import JSONLoader
from .parquet_loader import ParquetLoader


class DataLoaderFactory:
    """Factory for creating appropriate data loaders based on source type."""

    def __init__(self):
        """Initialize the data loader factory."""
        self.logger = logging.getLogger(__name__)

        # Register available loaders
        self._loaders: dict[str, type] = {
            "csv": CSVLoader,
            "tsv": CSVLoader,
            "txt": CSVLoader,
            "parquet": ParquetLoader,
            "pq": ParquetLoader,
            "json": JSONLoader,
            "jsonl": JSONLoader,
            "ndjson": JSONLoader,
            "xlsx": ExcelLoader,
            "xls": ExcelLoader,
        }

        # Default configurations for loaders
        self._default_configs: dict[str, dict[str, Any]] = {
            "csv": {"delimiter": ",", "encoding": "utf-8"},
            "tsv": {"delimiter": "\t", "encoding": "utf-8"},
            "txt": {"delimiter": ",", "encoding": "utf-8"},
            "parquet": {"engine": "pyarrow"},
            "json": {"orient": "records"},
            "excel": {"engine": "openpyxl"},
        }

    def create_loader(
        self,
        source: str | Path,
        loader_type: str | None = None,
        **kwargs: Any,
    ) -> DataLoaderProtocol:
        """Create appropriate data loader for the source.

        Args:
            source: Path to data source
            loader_type: Explicit loader type (auto-detected if None)
            **kwargs: Additional configuration for the loader

        Returns:
            Configured data loader instance

        Raises:
            DataValidationError: If no suitable loader found
        """
        # Auto-detect loader type if not provided
        if loader_type is None:
            loader_type = self._detect_loader_type(source)

        loader_type = loader_type.lower()

        if loader_type not in self._loaders:
            raise DataValidationError(
                f"Unsupported data format: {loader_type}",
                supported_formats=list(self._loaders.keys()),
            )

        # Get default configuration
        default_config = self._default_configs.get(loader_type, {})

        # Merge with user-provided kwargs
        config = {**default_config, **kwargs}

        # Create loader instance
        loader_class = self._loaders[loader_type]
        loader = loader_class(**config)

        self.logger.info(f"Created {loader_class.__name__} for {source}")

        return loader

    def load_data(
        self,
        source: str | Path,
        name: str | None = None,
        loader_type: str | None = None,
        **kwargs: Any,
    ) -> Dataset:
        """Load data using appropriate loader.

        Args:
            source: Path to data source
            name: Optional dataset name
            loader_type: Explicit loader type
            **kwargs: Additional loader configuration

        Returns:
            Loaded dataset
        """
        loader = self.create_loader(source, loader_type, **kwargs)
        return loader.load(source, name, **kwargs)

    def validate_source(
        self,
        source: str | Path,
        loader_type: str | None = None,
    ) -> bool:
        """Validate if source can be loaded.

        Args:
            source: Path to data source
            loader_type: Explicit loader type

        Returns:
            True if source is valid
        """
        try:
            loader = self.create_loader(source, loader_type)
            return loader.validate(source)
        except Exception:
            return False

    def get_supported_formats(self) -> list[str]:
        """Get list of supported file formats."""
        return list(self._loaders.keys())

    def get_loader_info(self, loader_type: str) -> dict[str, Any]:
        """Get information about a specific loader type.

        Args:
            loader_type: Type of loader

        Returns:
            Loader information
        """
        loader_type = loader_type.lower()

        if loader_type not in self._loaders:
            raise ValueError(f"Unknown loader type: {loader_type}")

        loader_class = self._loaders[loader_type]
        default_config = self._default_configs.get(loader_type, {})

        # Create temporary instance to get supported formats
        temp_loader = loader_class()

        return {
            "class_name": loader_class.__name__,
            "supported_formats": temp_loader.supported_formats,
            "default_config": default_config,
            "description": loader_class.__doc__ or "",
        }

    def register_loader(
        self,
        extensions: list[str],
        loader_class: type,
        default_config: dict[str, Any] | None = None,
    ) -> None:
        """Register a new data loader.

        Args:
            extensions: File extensions this loader handles
            loader_class: Loader class
            default_config: Default configuration for the loader
        """
        for ext in extensions:
            ext = ext.lower().lstrip(".")
            self._loaders[ext] = loader_class

            if default_config:
                self._default_configs[ext] = default_config

        self.logger.info(
            f"Registered {loader_class.__name__} for extensions: {extensions}"
        )

    def _detect_loader_type(self, source: str | Path) -> str:
        """Auto-detect the appropriate loader type for a source.

        Args:
            source: Path to data source

        Returns:
            Detected loader type

        Raises:
            DataValidationError: If loader type cannot be detected
        """
        source_path = Path(source)

        # Check if it's a URL
        if isinstance(source, str) and ("://" in source):
            parsed = urlparse(source)
            # Try to extract extension from URL path
            url_path = Path(parsed.path)
            if url_path.suffix:
                extension = url_path.suffix.lower().lstrip(".")
            else:
                raise DataValidationError(
                    f"Cannot detect file type from URL: {source}",
                    hint="Specify loader_type explicitly",
                )
        else:
            # Local file - use extension
            extension = source_path.suffix.lower().lstrip(".")

        if not extension:
            raise DataValidationError(
                f"Cannot detect file type: no extension found in {source}",
                hint="Specify loader_type explicitly",
            )

        if extension not in self._loaders:
            raise DataValidationError(
                f"Unsupported file extension: .{extension}",
                supported_formats=list(self._loaders.keys()),
            )

        return extension


class SmartDataLoader:
    """Smart data loader with automatic format detection and optimization."""

    def __init__(
        self,
        factory: DataLoaderFactory | None = None,
        auto_optimize: bool = True,
        memory_threshold_mb: float = 1000.0,
    ):
        """Initialize smart data loader.

        Args:
            factory: Data loader factory instance
            auto_optimize: Whether to automatically optimize loading strategy
            memory_threshold_mb: Memory threshold for batch loading
        """
        self.factory = factory or DataLoaderFactory()
        self.auto_optimize = auto_optimize
        self.memory_threshold_mb = memory_threshold_mb
        self.logger = logging.getLogger(__name__)

    def load(
        self,
        source: str | Path,
        name: str | None = None,
        **kwargs: Any,
    ) -> Dataset:
        """Smart load with automatic optimization.

        Args:
            source: Path to data source
            name: Optional dataset name
            **kwargs: Additional loading options

        Returns:
            Loaded dataset
        """
        source_path = Path(source)

        # Validate source exists
        if not source_path.exists():
            raise DataValidationError(f"Source file not found: {source}")

        # Get file size
        file_size_mb = source_path.stat().st_size / (1024 * 1024)
        self.logger.info(f"Loading {source} ({file_size_mb:.1f} MB)")

        # Create appropriate loader
        loader = self.factory.create_loader(source, **kwargs)

        # Decide loading strategy based on file size
        if self.auto_optimize and file_size_mb > self.memory_threshold_mb:
            return self._load_large_file(loader, source, name, **kwargs)
        else:
            return self._load_standard(loader, source, name, **kwargs)

    def load_multiple(
        self,
        sources: list[str | Path],
        names: list[str] | None = None,
        combine: bool = False,
        **kwargs: Any,
    ) -> list[Dataset] | Dataset:
        """Load multiple data sources.

        Args:
            sources: List of data sources
            names: Optional list of dataset names
            combine: Whether to combine into single dataset
            **kwargs: Additional loading options

        Returns:
            List of datasets or combined dataset
        """
        datasets = []

        for i, source in enumerate(sources):
            name = names[i] if names and i < len(names) else None
            dataset = self.load(source, name, **kwargs)
            datasets.append(dataset)

        if combine:
            return self._combine_datasets(datasets)

        return datasets

    def estimate_load_time(
        self,
        source: str | Path,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Estimate loading time and memory requirements.

        Args:
            source: Path to data source
            **kwargs: Loading options

        Returns:
            Estimation information
        """
        source_path = Path(source)

        if not source_path.exists():
            raise DataValidationError(f"Source file not found: {source}")

        # Get basic file info
        file_size_mb = source_path.stat().st_size / (1024 * 1024)

        # Try to get more detailed size estimation
        try:
            loader = self.factory.create_loader(source, **kwargs)

            if hasattr(loader, "estimate_size"):
                size_info = loader.estimate_size(source)
            else:
                size_info = {"file_size_mb": file_size_mb}

            # Estimate loading time (rough heuristic)
            # Assume ~50 MB/s loading speed for most formats
            estimated_load_time = file_size_mb / 50.0

            # Adjust based on format
            extension = source_path.suffix.lower()
            if extension in [".json", ".jsonl"]:
                estimated_load_time *= 2  # JSON is slower to parse
            elif extension in [".xlsx", ".xls"]:
                estimated_load_time *= 3  # Excel is much slower

            return {
                **size_info,
                "estimated_load_time_seconds": estimated_load_time,
                "recommended_batch_loading": file_size_mb > self.memory_threshold_mb,
                "file_extension": extension,
            }

        except Exception as e:
            return {
                "file_size_mb": file_size_mb,
                "error": str(e),
                "estimated_load_time_seconds": "unknown",
            }

    def _load_standard(
        self,
        loader: DataLoaderProtocol,
        source: str | Path,
        name: str | None,
        **kwargs: Any,
    ) -> Dataset:
        """Standard loading for smaller files."""
        return loader.load(source, name, **kwargs)

    def _load_large_file(
        self,
        loader: DataLoaderProtocol,
        source: str | Path,
        name: str | None,
        **kwargs: Any,
    ) -> Dataset:
        """Optimized loading for large files."""
        self.logger.info("Large file detected, using optimized loading strategy")

        # Check if loader supports batch loading
        if hasattr(loader, "load_batch"):
            # For very large files, we might want to load in batches
            # and combine, but for now, let's try standard loading with optimizations

            # Add memory-efficient options
            optimized_kwargs = kwargs.copy()
            optimized_kwargs.update(
                {
                    "low_memory": True,
                    "memory_map": True,  # For supported loaders
                }
            )

            try:
                return loader.load(source, name, **optimized_kwargs)
            except Exception as e:
                self.logger.warning(
                    f"Optimized loading failed: {e}, trying standard loading"
                )
                return loader.load(source, name, **kwargs)

        # Fallback to standard loading
        return loader.load(source, name, **kwargs)

    def _combine_datasets(self, datasets: list[Dataset]) -> Dataset:
        """Combine multiple datasets into one.

        Args:
            datasets: List of datasets to combine

        Returns:
            Combined dataset
        """
        if not datasets:
            raise ValueError("No datasets to combine")

        if len(datasets) == 1:
            return datasets[0]

        # Combine data
        combined_data = pd.concat([ds.data for ds in datasets], ignore_index=True)

        # Combine metadata
        combined_metadata = {
            "combined_from": [ds.name for ds in datasets],
            "original_shapes": [ds.data.shape for ds in datasets],
            "combined_shape": combined_data.shape,
        }

        # Merge individual metadata
        for i, ds in enumerate(datasets):
            combined_metadata[f"dataset_{i}_metadata"] = ds.metadata

        # Use target column from first dataset that has one
        target_column = None
        for ds in datasets:
            if ds.target_column:
                target_column = ds.target_column
                break

        return Dataset(
            name="combined_dataset",
            data=combined_data,
            target_column=target_column,
            metadata=combined_metadata,
        )


# Import pandas for combining datasets
import pandas as pd
