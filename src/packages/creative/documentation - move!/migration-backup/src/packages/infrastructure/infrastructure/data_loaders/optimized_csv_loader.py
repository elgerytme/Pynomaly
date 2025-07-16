"""Optimized CSV loader with memory efficiency and performance improvements."""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pynomaly.domain.entities import Dataset
from pynomaly.infrastructure.data_loaders.csv_loader import CSVLoader


class OptimizedCSVLoader(CSVLoader):
    """Memory-optimized CSV loader with intelligent type inference and chunked processing."""

    def __init__(
        self,
        chunk_size: int = 50000,
        memory_optimization: bool = True,
        dtype_inference: bool = True,
        categorical_threshold: float = 0.5,
    ):
        """Initialize optimized CSV loader.

        Args:
            chunk_size: Number of rows to process in each chunk
            memory_optimization: Enable aggressive memory optimization
            dtype_inference: Enable intelligent dtype inference
            categorical_threshold: Threshold for converting to categorical (ratio of unique values)
        """
        super().__init__()
        self.chunk_size = chunk_size
        self.memory_optimization = memory_optimization
        self.dtype_inference = dtype_inference
        self.categorical_threshold = categorical_threshold
        self.logger = logging.getLogger(__name__)

        # Cache for dtype optimizations
        self._dtype_cache: dict[str, dict[str, str]] = {}

    async def load(self, source: Path, **kwargs) -> Dataset:
        """Load CSV with memory and performance optimizations.

        Args:
            source: Path to CSV file
            **kwargs: Additional pandas read_csv arguments

        Returns:
            Optimized Dataset instance
        """
        self.logger.info(f"Loading CSV file: {source}")

        try:
            # Get file size for memory planning
            file_size_mb = source.stat().st_size / (1024 * 1024)
            self.logger.info(f"File size: {file_size_mb:.2f} MB")

            # Determine loading strategy based on file size
            if file_size_mb > 500:  # Large file - use chunked loading
                return await self._load_large_file(source, **kwargs)
            else:
                return await self._load_optimized(source, **kwargs)

        except Exception as e:
            self.logger.error(f"Failed to load CSV file {source}: {e}")
            raise

    async def _load_optimized(self, source: Path, **kwargs) -> Dataset:
        """Load file with dtype optimization."""
        # First pass: infer optimal dtypes
        if self.dtype_inference:
            sample_df = pd.read_csv(source, nrows=min(50000, self.chunk_size), **kwargs)
            optimal_dtypes = self._infer_optimal_dtypes(sample_df)

            # Cache dtypes for future use
            cache_key = f"{source.name}_{source.stat().st_mtime}"
            self._dtype_cache[cache_key] = optimal_dtypes
        else:
            optimal_dtypes = {}

        # Second pass: load with optimal types
        df = pd.read_csv(source, dtype=optimal_dtypes, **kwargs)

        # Apply memory optimizations
        if self.memory_optimization:
            original_memory = df.memory_usage(deep=True).sum()
            df = self._optimize_memory_usage(df)
            optimized_memory = df.memory_usage(deep=True).sum()

            memory_savings = (1 - optimized_memory / original_memory) * 100
            self.logger.info(f"Memory optimization: {memory_savings:.1f}% reduction")

        return Dataset(
            name=source.stem,
            data=df,
            metadata={
                "loader": "OptimizedCSVLoader",
                "file_path": str(source),
                "memory_optimized": self.memory_optimization,
                "dtype_optimized": self.dtype_inference,
                "original_size_mb": source.stat().st_size / (1024 * 1024),
                "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
                **self._get_optimization_metrics(df),
            },
        )

    async def _load_large_file(self, source: Path, **kwargs) -> Dataset:
        """Load large file using chunked processing."""
        self.logger.info(f"Using chunked loading for large file: {source}")

        # Read first chunk to infer dtypes and structure
        chunk_iter = pd.read_csv(source, chunksize=self.chunk_size, **kwargs)
        first_chunk = next(chunk_iter)

        if self.dtype_inference:
            optimal_dtypes = self._infer_optimal_dtypes(first_chunk)
        else:
            optimal_dtypes = {}

        # Process all chunks with optimized dtypes
        chunk_iter = pd.read_csv(
            source, chunksize=self.chunk_size, dtype=optimal_dtypes, **kwargs
        )

        chunks = []
        total_rows = 0

        for i, chunk in enumerate(chunk_iter):
            if self.memory_optimization:
                chunk = self._optimize_memory_usage(chunk)

            chunks.append(chunk)
            total_rows += len(chunk)

            if (i + 1) % 10 == 0:  # Log progress every 10 chunks
                self.logger.info(f"Processed {i + 1} chunks, {total_rows:,} rows")

                # Force garbage collection for large files
                if self.memory_optimization:
                    gc.collect()

        # Concatenate all chunks
        self.logger.info("Concatenating chunks...")
        df = pd.concat(chunks, ignore_index=True)

        # Final memory optimization
        if self.memory_optimization:
            df = self._optimize_memory_usage(df)
            gc.collect()

        return Dataset(
            name=source.stem,
            data=df,
            metadata={
                "loader": "OptimizedCSVLoader",
                "file_path": str(source),
                "chunked_loading": True,
                "chunk_size": self.chunk_size,
                "total_chunks": len(chunks),
                "memory_optimized": self.memory_optimization,
                "dtype_optimized": self.dtype_inference,
                "original_size_mb": source.stat().st_size / (1024 * 1024),
                "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
                **self._get_optimization_metrics(df),
            },
        )

    def _infer_optimal_dtypes(self, df: pd.DataFrame) -> dict[str, str]:
        """Infer optimal data types for memory efficiency."""
        dtypes = {}

        for col in df.columns:
            col_data = df[col]

            # Skip if already numeric
            if pd.api.types.is_numeric_dtype(col_data):
                if col_data.dtype == "int64":
                    dtypes[col] = self._optimize_integer_dtype(col_data)
                elif col_data.dtype == "float64":
                    dtypes[col] = "float32"  # Downcast to float32
                continue

            # Handle object columns
            if col_data.dtype == "object":
                optimized_dtype = self._optimize_object_dtype(col_data)
                if optimized_dtype:
                    dtypes[col] = optimized_dtype

        return dtypes

    def _optimize_integer_dtype(self, series: pd.Series) -> str:
        """Optimize integer column dtype."""
        min_val = series.min()
        max_val = series.max()

        # Check for unsigned integers
        if min_val >= 0:
            if max_val <= 255:
                return "uint8"
            elif max_val <= 65535:
                return "uint16"
            elif max_val <= 4294967295:
                return "uint32"
        else:
            # Signed integers
            if min_val >= -128 and max_val <= 127:
                return "int8"
            elif min_val >= -32768 and max_val <= 32767:
                return "int16"
            elif min_val >= -2147483648 and max_val <= 2147483647:
                return "int32"

        return "int64"  # Keep original if no optimization possible

    def _optimize_object_dtype(self, series: pd.Series) -> str | None:
        """Optimize object column dtype."""
        # Try numeric conversion first
        try:
            pd.to_numeric(series, errors="raise")
            return "float32"  # Use float32 for numeric data
        except (ValueError, TypeError):
            pass

        # Try datetime conversion
        try:
            pd.to_datetime(series, errors="raise")
            return None  # Let pandas handle datetime inference
        except (ValueError, TypeError):
            pass

        # Check for categorical data
        unique_ratio = series.nunique() / len(series)
        if unique_ratio < self.categorical_threshold:
            return "category"

        return None  # Keep as object

    def _optimize_memory_usage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply comprehensive memory optimizations."""
        optimized_df = df.copy()

        for col in optimized_df.columns:
            col_data = optimized_df[col]

            # Optimize numeric columns
            if pd.api.types.is_integer_dtype(col_data):
                optimized_df[col] = pd.to_numeric(col_data, downcast="integer")
            elif pd.api.types.is_float_dtype(col_data):
                optimized_df[col] = pd.to_numeric(col_data, downcast="float")

            # Convert to categorical if beneficial
            elif col_data.dtype == "object":
                unique_ratio = col_data.nunique() / len(col_data)
                if unique_ratio < self.categorical_threshold:
                    optimized_df[col] = col_data.astype("category")

        return optimized_df

    def _get_optimization_metrics(self, df: pd.DataFrame) -> dict[str, Any]:
        """Get metrics about the optimization process."""
        metrics = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "numeric_columns": len(df.select_dtypes(include=[np.number]).columns),
            "categorical_columns": len(df.select_dtypes(include=["category"]).columns),
            "object_columns": len(df.select_dtypes(include=["object"]).columns),
            "datetime_columns": len(df.select_dtypes(include=["datetime"]).columns),
        }

        # Memory usage by dtype
        memory_by_dtype = {}
        for dtype in df.dtypes.unique():
            cols_of_dtype = df.select_dtypes(include=[dtype]).columns
            if len(cols_of_dtype) > 0:
                memory_usage = df[cols_of_dtype].memory_usage(deep=True).sum()
                memory_by_dtype[str(dtype)] = memory_usage / (1024 * 1024)  # MB

        metrics["memory_usage_by_dtype_mb"] = memory_by_dtype

        return metrics


class ParallelCSVLoader:
    """Parallel CSV loader for processing multiple files simultaneously."""

    def __init__(
        self,
        max_workers: int = 4,
        optimized_loader_config: dict[str, Any] | None = None,
    ):
        """Initialize parallel CSV loader.

        Args:
            max_workers: Maximum number of worker threads
            optimized_loader_config: Configuration for OptimizedCSVLoader
        """
        self.max_workers = max_workers
        self.optimized_loader_config = optimized_loader_config or {}
        self.logger = logging.getLogger(__name__)

    async def load_multiple_files(self, file_paths: list[Path]) -> list[Dataset]:
        """Load multiple CSV files in parallel.

        Args:
            file_paths: List of CSV file paths to load

        Returns:
            List of Dataset instances
        """
        import asyncio
        from concurrent.futures import ThreadPoolExecutor

        self.logger.info(
            f"Loading {len(file_paths)} files in parallel with {self.max_workers} workers"
        )

        # Create optimized loader
        loader = OptimizedCSVLoader(**self.optimized_loader_config)

        # Create tasks for parallel execution
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            loop = asyncio.get_event_loop()

            tasks = [
                loop.run_in_executor(
                    executor, self._load_single_file, loader, file_path
                )
                for file_path in file_paths
            ]

            # Wait for all tasks to complete
            datasets = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter out exceptions and log errors
            successful_datasets = []
            for i, result in enumerate(datasets):
                if isinstance(result, Exception):
                    self.logger.error(f"Failed to load {file_paths[i]}: {result}")
                else:
                    successful_datasets.append(result)

            self.logger.info(
                f"Successfully loaded {len(successful_datasets)}/{len(file_paths)} files"
            )
            return successful_datasets

    def _load_single_file(self, loader: OptimizedCSVLoader, file_path: Path) -> Dataset:
        """Load single file (runs in thread)."""
        import asyncio

        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            return loop.run_until_complete(loader.load(file_path))
        finally:
            loop.close()


# Performance monitoring for data loading
class DataLoadingProfiler:
    """Profiler for monitoring data loading performance."""

    def __init__(self):
        self.metrics = {}
        self.logger = logging.getLogger(__name__)

    def profile_loading(self, func):
        """Decorator to profile data loading functions."""
        import functools
        import time

        import psutil

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB

            try:
                result = await func(*args, **kwargs)

                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB

                # Record metrics
                duration = end_time - start_time
                memory_delta = end_memory - start_memory

                self.metrics[func.__name__] = {
                    "duration_seconds": duration,
                    "memory_delta_mb": memory_delta,
                    "start_memory_mb": start_memory,
                    "end_memory_mb": end_memory,
                    "timestamp": start_time,
                }

                self.logger.info(
                    f"{func.__name__} completed in {duration:.2f}s, "
                    f"memory delta: {memory_delta:+.2f} MB"
                )

                return result

            except Exception as e:
                self.logger.error(f"{func.__name__} failed: {e}")
                raise

        return wrapper

    def get_performance_summary(self) -> dict[str, Any]:
        """Get summary of performance metrics."""
        if not self.metrics:
            return {"message": "No metrics available"}

        total_time = sum(m["duration_seconds"] for m in self.metrics.values())
        total_memory = sum(m["memory_delta_mb"] for m in self.metrics.values())

        return {
            "total_operations": len(self.metrics),
            "total_time_seconds": total_time,
            "total_memory_delta_mb": total_memory,
            "average_time_seconds": total_time / len(self.metrics),
            "average_memory_delta_mb": total_memory / len(self.metrics),
            "operations": self.metrics,
        }
