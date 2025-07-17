"""Unified data service integrating loading, processing, and validation."""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd

from monorepo.domain.entities import Dataset
from monorepo.domain.exceptions import DataValidationError
from monorepo.infrastructure.data_loaders.data_loader_factory import (
    DataLoaderFactory,
    SmartDataLoader,
)
from monorepo.infrastructure.data_loaders.database_loader import DatabaseLoader
from monorepo.infrastructure.data_processing.advanced_data_pipeline import (
    AdvancedDataPipeline,
    ProcessingConfig,
    ProcessingReport,
)


class DataSourceType:
    """Data source type constants."""

    FILE = "file"
    DATABASE = "database"
    URL = "url"
    DATAFRAME = "dataframe"
    STREAM = "stream"


class UnifiedDataService:
    """Unified service for data loading, processing, and validation."""

    def __init__(
        self,
        data_loader_factory: DataLoaderFactory | None = None,
        database_loader: DatabaseLoader | None = None,
        data_pipeline: AdvancedDataPipeline | None = None,
        max_workers: int = 4,
        default_processing_config: ProcessingConfig | None = None,
    ):
        """Initialize unified data service.

        Args:
            data_loader_factory: Factory for creating data loaders
            database_loader: Database loader instance
            data_pipeline: Data processing pipeline
            max_workers: Maximum number of parallel workers
            default_processing_config: Default processing configuration
        """
        self.data_loader_factory = data_loader_factory or DataLoaderFactory()
        self.database_loader = database_loader or DatabaseLoader()
        self.data_pipeline = data_pipeline or AdvancedDataPipeline()
        self.max_workers = max_workers
        self.default_processing_config = default_processing_config or ProcessingConfig()

        self.logger = logging.getLogger(__name__)

        # Smart loader for optimization
        self.smart_loader = SmartDataLoader(
            factory=self.data_loader_factory, auto_optimize=True
        )

        # Processing history and caching
        self._processing_cache: dict[str, tuple[DataCollection, ProcessingReport]] = {}
        self._data_collection_registry: dict[str, DataCollection] = {}

    async def load_and_process(
        self,
        source: str | Path | pd.DataFrame,
        name: str | None = None,
        processing_config: ProcessingConfig | None = None,
        auto_detect_target: bool = True,
        return_report: bool = False,
        cache_result: bool = True,
        **kwargs: Any,
    ) -> DataCollection | tuple[DataCollection, ProcessingReport]:
        """Load data from source and apply processing pipeline.

        Args:
            source: Data source (file, database, DataFrame, etc.)
            name: Optional data_collection name
            processing_config: Processing configuration
            auto_detect_target: Whether to auto-detect target column
            return_report: Whether to return processing report
            cache_result: Whether to cache the result
            **kwargs: Additional loading/processing options

        Returns:
            Processed data_collection, optionally with processing report
        """
        start_time = time.perf_counter()

        # Load data
        raw_data_collection = await self._load_data_async(source, name, **kwargs)

        # Auto-detect target column if requested
        if auto_detect_target and not raw_data_collection.target_column:
            target_column = self._auto_detect_target_column(raw_data_collection.data)
            if target_column:
                raw_data_collection.target_column = target_column
                self.logger.info(f"Auto-detected target column: {target_column}")

        # Apply processing pipeline
        config = processing_config or self.default_processing_config

        # Check cache first
        cache_key = self._generate_cache_key(raw_data_collection, config)
        if cache_result and cache_key in self._processing_cache:
            self.logger.info("Using cached processing result")
            cached_result = self._processing_cache[cache_key]
            if return_report:
                return cached_result
            return cached_result[0]

        # Configure pipeline
        pipeline = AdvancedDataPipeline(config=config)

        # Process data
        if return_report:
            processed_data_collection, report = pipeline.process_data_collection(
                raw_data_collection, fit_transformers=True, return_report=True
            )
        else:
            processed_data_collection = pipeline.process_data_collection(
                raw_data_collection, fit_transformers=True, return_report=False
            )
            report = None

        # Cache result
        if cache_result:
            if report:
                self._processing_cache[cache_key] = (processed_data_collection, report)
            else:
                # Create a basic report
                basic_report = ProcessingReport(
                    original_shape=raw_data_collection.data.shape,
                    final_shape=processed_data_collection.data.shape,
                    processing_time=time.perf_counter() - start_time,
                    steps_performed=["unified_processing"],
                )
                self._processing_cache[cache_key] = (processed_data_collection, basic_report)

        # Register data_collection
        self._data_collection_registry[processed_data_collection.name] = processed_data_collection

        self.logger.info(
            f"Completed load and process in {time.perf_counter() - start_time:.2f}s"
        )

        if return_report:
            return processed_data_collection, report
        return processed_data_collection

    async def load_multiple_sources(
        self,
        sources: list[str | Path | pd.DataFrame],
        names: list[str] | None = None,
        combine: bool = False,
        processing_config: ProcessingConfig | None = None,
        parallel: bool = True,
        **kwargs: Any,
    ) -> list[DataCollection] | DataCollection:
        """Load and process multiple data sources.

        Args:
            sources: List of data sources
            names: Optional list of data_collection names
            combine: Whether to combine datasets into one
            processing_config: Processing configuration
            parallel: Whether to process in parallel
            **kwargs: Additional options

        Returns:
            List of datasets or combined data_collection
        """
        self.logger.info(f"Loading {len(sources)} data sources")

        if parallel and len(sources) > 1:
            # Process in parallel
            tasks = []
            for i, source in enumerate(sources):
                name = names[i] if names and i < len(names) else None
                task = self.load_and_process(source, name, processing_config, **kwargs)
                tasks.append(task)

            datasets = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter out exceptions
            valid_data_collections = [ds for ds in datasets if isinstance(ds, DataCollection)]

            # Log any failures
            for i, result in enumerate(datasets):
                if isinstance(result, Exception):
                    self.logger.error(f"Failed to load source {i}: {result}")

        else:
            # Process sequentially
            valid_data_collections = []
            for i, source in enumerate(sources):
                try:
                    name = names[i] if names and i < len(names) else None
                    data_collection = await self.load_and_process(
                        source, name, processing_config, **kwargs
                    )
                    valid_data_collections.append(data_collection)
                except Exception as e:
                    self.logger.error(f"Failed to load source {i}: {e}")
                    continue

        if not valid_data_collections:
            raise DataValidationError("No sources could be loaded successfully")

        if combine:
            return self._combine_data_collections(valid_data_collections)

        return valid_data_collections

    def create_processing_config(
        self,
        data_collection_characteristics: dict[str, Any] | None = None,
        use_case: str = "anomaly_processing",
        performance_preference: str = "balanced",
        **custom_settings: Any,
    ) -> ProcessingConfig:
        """Create optimized processing configuration.

        Args:
            data_collection_characteristics: DataCollection characteristics for optimization
            use_case: Use case type (anomaly_processing, classification, etc.)
            performance_preference: Performance preference (fast, balanced, thorough)
            **custom_settings: Custom configuration overrides

        Returns:
            Optimized processing configuration
        """
        # Start with base configuration
        config = ProcessingConfig()

        # Optimize based on data_collection characteristics
        if data_collection_characteristics:
            n_samples = data_collection_characteristics.get("n_samples", 0)
            n_features = data_collection_characteristics.get("n_features", 0)
            has_categorical = data_collection_characteristics.get("has_categorical", False)

            # Adjust for large datasets
            if n_samples > 100000:
                config.memory_efficient = True
                config.parallel_processing = True
                config.max_workers = min(self.max_workers, 8)

            # Adjust for high-dimensional data
            if n_features > 1000:
                config.apply_feature_selection = True
                config.max_features = min(500, n_features // 2)
                config.remove_low_variance = True
                config.variance_threshold = 0.05

            # Handle categorical features
            if has_categorical:
                config.encode_categoricals = True
                config.max_categories = 20 if n_samples > 10000 else 10

        # Optimize for use case
        if use_case == "anomaly_processing":
            # Anomaly processing specific optimizations
            config.apply_scaling = True
            config.scaling_method = config.scaling_method  # Keep robust scaling
            config.handle_missing = True
            config.remove_duplicates = True

        elif use_case == "classification":
            # Classification specific optimizations
            config.apply_feature_selection = True
            config.scale_target = False  # Don't scale classification targets

        # Optimize for performance preference
        if performance_preference == "fast":
            config.parallel_processing = True
            config.memory_efficient = True
            config.validate_data = False
            config.apply_feature_selection = False

        elif performance_preference == "thorough":
            config.validate_data = True
            config.strict_validation = True
            config.apply_feature_selection = True
            config.remove_low_variance = True

        # Apply custom settings
        for key, value in custom_settings.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return config

    def validate_dataset_quality(
        self,
        data_collection: DataCollection,
        requirements: dict[str, Any] | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        """Validate data_collection quality for anomaly processing.

        Args:
            data_collection: DataCollection to validate
            requirements: Quality requirements

        Returns:
            Tuple of (is_valid, quality_report)
        """
        requirements = requirements or {}

        data = data_collection.data
        quality_report = {
            "overall_quality": "unknown",
            "issues": [],
            "warnings": [],
            "recommendations": [],
            "measurements": {},
        }

        # Basic measurements
        n_samples, n_features = data.shape
        missing_percentage = data.isnull().mean().mean()
        duplicate_rows = data.duplicated().sum()

        quality_report["measurements"] = {
            "n_samples": n_samples,
            "n_features": n_features,
            "missing_percentage": missing_percentage,
            "duplicate_rows": duplicate_rows,
            "memory_usage_mb": data.memory_usage(deep=True).sum() / (1024 * 1024),
        }

        # Quality checks
        issues = []
        warnings = []
        recommendations = []

        # Check minimum sample size
        min_samples = requirements.get("min_samples", 100)
        if n_samples < min_samples:
            issues.append(f"Insufficient samples: {n_samples} < {min_samples}")

        # Check maximum missing values
        max_missing = requirements.get("max_missing_percentage", 0.5)
        if missing_percentage > max_missing:
            issues.append(
                f"Too many missing values: {missing_percentage:.2%} > {max_missing:.2%}"
            )

        # Check for excessive duplicates
        max_duplicates = requirements.get("max_duplicate_percentage", 0.1)
        duplicate_percentage = duplicate_rows / n_samples
        if duplicate_percentage > max_duplicates:
            warnings.append(f"High duplicate rate: {duplicate_percentage:.2%}")

        # Check feature variability
        numeric_cols = data.select_dtypes(include=["number"]).columns
        if len(numeric_cols) > 0:
            low_variance_cols = []
            for col in numeric_cols:
                if data[col].var() < 1e-8:
                    low_variance_cols.append(col)

            if low_variance_cols:
                warnings.append(f"Low variance features: {low_variance_cols}")
                recommendations.append("Consider removing low variance features")

        # Check for highly correlated features
        if len(numeric_cols) > 1:
            corr_matrix = data[numeric_cols].corr().abs()
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > 0.95:
                        high_corr_pairs.append(
                            (corr_matrix.columns[i], corr_matrix.columns[j])
                        )

            if high_corr_pairs:
                warnings.append(f"Highly correlated features: {high_corr_pairs}")
                recommendations.append("Consider removing redundant features")

        # Memory usage check
        max_memory_mb = requirements.get("max_memory_mb", 1000)
        memory_usage = quality_report["measurements"]["memory_usage_mb"]
        if memory_usage > max_memory_mb:
            warnings.append(f"Large memory usage: {memory_usage:.1f} MB")
            recommendations.append("Consider batch processing or data reduction")

        # Determine overall quality
        if issues:
            quality_report["overall_quality"] = "poor"
        elif warnings:
            quality_report["overall_quality"] = "fair"
        else:
            quality_report["overall_quality"] = "good"

        quality_report["issues"] = issues
        quality_report["warnings"] = warnings
        quality_report["recommendations"] = recommendations

        is_valid = len(issues) == 0
        return is_valid, quality_report

    def get_dataset_summary(self, dataset_name: str) -> dict[str, Any] | None:
        """Get summary information about a registered data_collection.

        Args:
            data_collection_name: Name of the data_collection

        Returns:
            DataCollection summary or None if not found
        """
        if data_collection_name not in self._data_collection_registry:
            return None

        data_collection = self._data_collection_registry[data_collection_name]
        data = data_collection.data

        # Basic information
        summary = {
            "name": data_collection.name,
            "shape": data.shape,
            "target_column": data_collection.target_column,
            "has_target": data_collection.has_target,
            "memory_usage_mb": data.memory_usage(deep=True).sum() / (1024 * 1024),
        }

        # Column information
        summary["columns"] = {
            "total": len(data.columns),
            "numeric": len(data.select_dtypes(include=["number"]).columns),
            "categorical": len(
                data.select_dtypes(include=["object", "category"]).columns
            ),
            "datetime": len(data.select_dtypes(include=["datetime64"]).columns),
        }

        # Data quality measurements
        summary["quality"] = {
            "missing_values": data.isnull().sum().sum(),
            "missing_percentage": data.isnull().mean().mean(),
            "duplicate_rows": data.duplicated().sum(),
            "unique_rows": len(data.drop_duplicates()),
        }

        # Metadata
        summary["metadata"] = data_collection.metadata

        return summary

    def list_registered_datasets(self) -> list[str]:
        """Get list of all registered data_collection names."""
        return list(self._data_collection_registry.keys())

    def get_registered_dataset(self, name: str) -> Dataset | None:
        """Get a registered data_collection by name."""
        return self._data_collection_registry.get(name)

    def clear_cache(self) -> None:
        """Clear all caches."""
        self._processing_cache.clear()
        self._data_collection_registry.clear()
        self.logger.info("Cleared all caches")

    async def _load_data_async(
        self,
        source: str | Path | pd.DataFrame,
        name: str | None,
        **kwargs: Any,
    ) -> DataCollection:
        """Load data asynchronously."""
        loop = asyncio.get_event_loop()

        with ThreadPoolExecutor(max_workers=1) as executor:
            return await loop.run_in_executor(
                executor, self._load_data_sync, source, name, kwargs
            )

    def _load_data_sync(
        self,
        source: str | Path | pd.DataFrame,
        name: str | None,
        kwargs: dict[str, Any],
    ) -> DataCollection:
        """Load data synchronously."""
        # Handle different source types
        if isinstance(source, pd.DataFrame):
            return self._load_from_dataframe(source, name, **kwargs)

        elif isinstance(source, (str, Path)):
            source_str = str(source)

            # Check if it's a database connection string
            if self._is_database_source(source_str):
                return self._load_from_database(source_str, name, **kwargs)

            # Check if it's a URL
            elif source_str.startswith(("http://", "https://", "ftp://")):
                return self._load_from_url(source_str, name, **kwargs)

            # Assume it's a file
            else:
                return self._load_from_file(source, name, **kwargs)

        else:
            raise DataValidationError(f"Unsupported source type: {type(source)}")

    def _load_from_dataframe(
        self, df: pd.DataFrame, name: str | None, **kwargs: Any
    ) -> DataCollection:
        """Load data from pandas DataFrame."""
        data_collection_name = name or "dataframe_data_collection"

        if df.empty:
            raise DataValidationError("DataFrame is empty")

        return DataCollection(
            name=data_collection_name,
            data=df.copy(),
            target_column=kwargs.get("target_column"),
            metadata={
                "source": "dataframe",
                "loader": "UnifiedDataService",
                "original_shape": df.shape,
            },
        )

    def _load_from_file(
        self, file_path: str | Path, name: str | None, **kwargs: Any
    ) -> DataCollection:
        """Load data from file."""
        return self.smart_loader.load(file_path, name, **kwargs)

    def _load_from_database(
        self, connection_string: str, name: str | None, **kwargs: Any
    ) -> DataCollection:
        """Load data from database."""
        query = kwargs.get("query")
        table_name = kwargs.get("table_name")

        if query:
            return self.database_loader.load_query(
                query, connection_string, name, **kwargs
            )
        elif table_name:
            return self.database_loader.load_table(
                table_name, connection_string, name=name, **kwargs
            )
        else:
            raise DataValidationError(
                "Either 'query' or 'table_name' must be provided for database sources"
            )

    def _load_from_url(self, url: str, name: str | None, **kwargs: Any) -> Dataset:
        """Load data from URL."""
        # For now, delegate to smart loader
        # In a full implementation, you might want special URL handling
        return self.smart_loader.load(url, name, **kwargs)

    def _is_database_source(self, source: str) -> bool:
        """Check if source is a database connection string."""
        db_schemes = ["postgresql", "mysql", "sqlite", "mssql", "oracle", "snowflake"]
        return any(source.startswith(f"{scheme}://") for scheme in db_schemes)

    def _auto_detect_target_column(self, data: pd.DataFrame) -> str | None:
        """Auto-detect potential target column."""
        # Common target column names for anomaly processing
        target_candidates = [
            "target",
            "label",
            "class",
            "anomaly",
            "outlier",
            "is_anomaly",
            "is_outlier",
            "y",
            "outcome",
        ]

        for col in data.columns:
            col_lower = col.lower()
            if col_lower in target_candidates:
                return col

        # Check for binary columns that might be targets
        for col in data.columns:
            if data[col].dtype in ["int64", "float64", "bool"]:
                unique_values = data[col].dropna().unique()
                if len(unique_values) == 2 and set(unique_values).issubset({0, 1}):
                    return col

        return None

    def _combine_datasets(self, datasets: list[Dataset]) -> Dataset:
        """Combine multiple datasets into one."""
        if not datasets:
            raise ValueError("No datasets to combine")

        if len(datasets) == 1:
            return datasets[0]

        # Combine data
        combined_data = pd.concat([ds.data for ds in datasets], ignore_index=True)

        # Use first data_collection's target column
        target_column = None
        for ds in datasets:
            if ds.target_column:
                target_column = ds.target_column
                break

        # Combine metadata
        combined_metadata = {
            "combined_from": [ds.name for ds in datasets],
            "original_shapes": [ds.data.shape for ds in datasets],
            "combined_shape": combined_data.shape,
            "source": "combined_data_collections",
            "loader": "UnifiedDataService",
        }

        return DataCollection(
            name="combined_data_collection",
            data=combined_data,
            target_column=target_column,
            metadata=combined_metadata,
        )

    def _generate_cache_key(self, dataset: Dataset, config: ProcessingConfig) -> str:
        """Generate cache key for processing results."""
        # Create a hash of data_collection and config
        import hashlib

        data_hash = hashlib.md5(
            str(data_collection.data.values.tobytes()).encode()
        ).hexdigest()[:8]
        config_hash = hashlib.md5(str(config.__dict__).encode()).hexdigest()[:8]

        return f"{data_collection.name}_{data_hash}_{config_hash}"
