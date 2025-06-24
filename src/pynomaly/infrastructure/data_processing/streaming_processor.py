"""Memory-efficient data processing with streaming capabilities.

This module provides streaming data processing for large datasets while
maintaining the simplified architecture principles established in Phase 1.
"""

from __future__ import annotations

import gc
import warnings
from contextlib import contextmanager
from typing import Any, Dict, Generator, Iterator, Optional, Protocol, Union

import numpy as np
import pandas as pd

from ...domain.entities import Dataset
from ...infrastructure.config.feature_flags import require_feature


class DataChunk(Protocol):
    """Protocol for data chunks in streaming processing."""
    
    def __len__(self) -> int:
        """Get the number of rows in the chunk."""
        ...
    
    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array."""
        ...


class StreamingDataProcessor:
    """Memory-efficient streaming data processor."""
    
    def __init__(
        self,
        chunk_size: int = 10000,
        memory_limit_mb: int = 500,
        enable_gc: bool = True
    ):
        """Initialize streaming processor.
        
        Args:
            chunk_size: Number of rows per chunk
            memory_limit_mb: Memory limit in MB for processing
            enable_gc: Whether to enable aggressive garbage collection
        """
        self.chunk_size = chunk_size
        self.memory_limit_mb = memory_limit_mb
        self.enable_gc = enable_gc
        self._processed_chunks = 0
        self._total_memory_used = 0.0
    
    @require_feature("memory_efficiency")
    def process_large_dataset(
        self,
        data_source: Union[str, pd.DataFrame, np.ndarray],
        transform_func: Optional[callable] = None,
        **kwargs
    ) -> Generator[Dataset, None, None]:
        """Process large dataset in memory-efficient chunks.
        
        Args:
            data_source: Path to file or data object
            transform_func: Optional transformation function
            **kwargs: Additional arguments for data loading
            
        Yields:
            Dataset chunks for processing
        """
        chunk_iterator = self._create_chunk_iterator(data_source, **kwargs)
        
        for chunk_data in chunk_iterator:
            with self._memory_management():
                # Apply transformation if provided
                if transform_func:
                    chunk_data = transform_func(chunk_data)
                
                # Create dataset chunk
                chunk_dataset = Dataset(
                    name=f"chunk_{self._processed_chunks}",
                    data=chunk_data if isinstance(chunk_data, np.ndarray) else chunk_data.values
                )
                
                self._processed_chunks += 1
                yield chunk_dataset
    
    @require_feature("memory_efficiency")
    def batch_process_files(
        self,
        file_paths: list[str],
        file_type: str = "auto"
    ) -> Generator[Dataset, None, None]:
        """Process multiple files in batches to manage memory."""
        for file_path in file_paths:
            try:
                # Determine file type if auto
                if file_type == "auto":
                    file_type = self._detect_file_type(file_path)
                
                # Process file in chunks
                yield from self.process_large_dataset(file_path)
                
            except Exception as e:
                warnings.warn(f"Failed to process {file_path}: {e}")
                continue
    
    @require_feature("memory_efficiency")
    def aggregate_streaming_results(
        self,
        chunk_results: Iterator[Any],
        aggregation_func: callable
    ) -> Any:
        """Aggregate results from streaming processing."""
        accumulated = None
        
        for result in chunk_results:
            with self._memory_management():
                if accumulated is None:
                    accumulated = result
                else:
                    accumulated = aggregation_func(accumulated, result)
        
        return accumulated
    
    def _create_chunk_iterator(
        self,
        data_source: Union[str, pd.DataFrame, np.ndarray],
        **kwargs
    ) -> Iterator[Union[pd.DataFrame, np.ndarray]]:
        """Create appropriate chunk iterator for data source."""
        if isinstance(data_source, str):
            # File path
            return self._file_chunk_iterator(data_source, **kwargs)
        elif isinstance(data_source, pd.DataFrame):
            # In-memory DataFrame
            return self._dataframe_chunk_iterator(data_source)
        elif isinstance(data_source, np.ndarray):
            # In-memory numpy array
            return self._numpy_chunk_iterator(data_source)
        else:
            raise ValueError(f"Unsupported data source type: {type(data_source)}")
    
    def _file_chunk_iterator(
        self,
        file_path: str,
        **kwargs
    ) -> Iterator[pd.DataFrame]:
        """Create iterator for file-based chunks."""
        file_type = self._detect_file_type(file_path)
        
        if file_type == "csv":
            yield from pd.read_csv(
                file_path,
                chunksize=self.chunk_size,
                **kwargs
            )
        elif file_type == "parquet":
            # For parquet, we need to read in chunks manually
            df = pd.read_parquet(file_path, **kwargs)
            yield from self._dataframe_chunk_iterator(df)
        elif file_type == "excel":
            df = pd.read_excel(file_path, **kwargs)
            yield from self._dataframe_chunk_iterator(df)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    
    def _dataframe_chunk_iterator(
        self,
        df: pd.DataFrame
    ) -> Iterator[pd.DataFrame]:
        """Create iterator for DataFrame chunks."""
        for start in range(0, len(df), self.chunk_size):
            end = min(start + self.chunk_size, len(df))
            yield df.iloc[start:end].copy()
    
    def _numpy_chunk_iterator(
        self,
        array: np.ndarray
    ) -> Iterator[np.ndarray]:
        """Create iterator for numpy array chunks."""
        for start in range(0, len(array), self.chunk_size):
            end = min(start + self.chunk_size, len(array))
            yield array[start:end].copy()
    
    def _detect_file_type(self, file_path: str) -> str:
        """Detect file type from extension."""
        file_path = file_path.lower()
        if file_path.endswith('.csv'):
            return "csv"
        elif file_path.endswith('.parquet'):
            return "parquet"
        elif file_path.endswith(('.xls', '.xlsx')):
            return "excel"
        elif file_path.endswith('.json'):
            return "json"
        else:
            return "csv"  # Default fallback
    
    @contextmanager
    def _memory_management(self):
        """Context manager for memory management during processing."""
        import psutil
        
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        try:
            yield
        finally:
            if self.enable_gc:
                gc.collect()
            
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            chunk_memory = memory_after - memory_before
            self._total_memory_used += chunk_memory
            
            # Warn if approaching memory limit
            if memory_after > self.memory_limit_mb * 0.8:
                warnings.warn(
                    f"Memory usage ({memory_after:.1f}MB) approaching limit "
                    f"({self.memory_limit_mb}MB). Consider reducing chunk size."
                )


class MemoryOptimizedDataLoader:
    """Data loader optimized for memory efficiency."""
    
    def __init__(self, processor: Optional[StreamingDataProcessor] = None):
        """Initialize memory-optimized data loader."""
        self.processor = processor or StreamingDataProcessor()
    
    @require_feature("memory_efficiency")
    def load_dataset_efficiently(
        self,
        data_source: Union[str, pd.DataFrame, np.ndarray],
        target_memory_mb: Optional[int] = None
    ) -> Dataset:
        """Load dataset with memory optimization."""
        if target_memory_mb:
            self.processor.memory_limit_mb = target_memory_mb
        
        # For small datasets, load directly
        if self._is_small_dataset(data_source):
            return self._load_direct(data_source)
        
        # For large datasets, use streaming approach
        return self._load_streaming(data_source)
    
    @require_feature("memory_efficiency")
    def optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage by downcasting types."""
        optimized_df = df.copy()
        
        # Optimize integer columns
        for col in optimized_df.select_dtypes(include=['int']).columns:
            col_data = optimized_df[col]
            if col_data.min() >= 0:
                # Unsigned integers
                if col_data.max() < 256:
                    optimized_df[col] = col_data.astype('uint8')
                elif col_data.max() < 65536:
                    optimized_df[col] = col_data.astype('uint16')
                elif col_data.max() < 4294967296:
                    optimized_df[col] = col_data.astype('uint32')
            else:
                # Signed integers
                if col_data.min() >= -128 and col_data.max() <= 127:
                    optimized_df[col] = col_data.astype('int8')
                elif col_data.min() >= -32768 and col_data.max() <= 32767:
                    optimized_df[col] = col_data.astype('int16')
                elif col_data.min() >= -2147483648 and col_data.max() <= 2147483647:
                    optimized_df[col] = col_data.astype('int32')
        
        # Optimize float columns
        for col in optimized_df.select_dtypes(include=['float']).columns:
            optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='float')
        
        # Optimize object columns to category where appropriate
        for col in optimized_df.select_dtypes(include=['object']).columns:
            num_unique = optimized_df[col].nunique()
            total_count = len(optimized_df[col])
            
            # Convert to category if less than 50% unique values
            if num_unique / total_count < 0.5:
                optimized_df[col] = optimized_df[col].astype('category')
        
        return optimized_df
    
    def _is_small_dataset(self, data_source: Union[str, pd.DataFrame, np.ndarray]) -> bool:
        """Check if dataset is small enough for direct loading."""
        if isinstance(data_source, str):
            # Estimate file size
            import os
            file_size_mb = os.path.getsize(data_source) / 1024 / 1024
            return file_size_mb < 50  # 50MB threshold
        elif isinstance(data_source, pd.DataFrame):
            return data_source.memory_usage(deep=True).sum() / 1024 / 1024 < 50
        elif isinstance(data_source, np.ndarray):
            return data_source.nbytes / 1024 / 1024 < 50
        return False
    
    def _load_direct(self, data_source: Union[str, pd.DataFrame, np.ndarray]) -> Dataset:
        """Load dataset directly into memory."""
        if isinstance(data_source, str):
            # Load file
            file_type = self.processor._detect_file_type(data_source)
            if file_type == "csv":
                data = pd.read_csv(data_source)
            elif file_type == "parquet":
                data = pd.read_parquet(data_source)
            elif file_type == "excel":
                data = pd.read_excel(data_source)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            # Optimize memory
            data = self.optimize_dataframe_memory(data)
            data_array = data.values
            
        elif isinstance(data_source, pd.DataFrame):
            optimized_df = self.optimize_dataframe_memory(data_source)
            data_array = optimized_df.values
        else:
            data_array = data_source
        
        return Dataset(
            name="optimized_dataset",
            data=data_array
        )
    
    def _load_streaming(self, data_source: Union[str, pd.DataFrame, np.ndarray]) -> Dataset:
        """Load dataset using streaming approach."""
        # For streaming, we'll load the first chunk as a sample
        # In a real implementation, this might return a special streaming dataset
        chunk_generator = self.processor.process_large_dataset(data_source)
        first_chunk = next(chunk_generator)
        
        # Add metadata about streaming nature
        first_chunk.name = "streaming_dataset_sample"
        
        return first_chunk


class LargeDatasetAnalyzer:
    """Analyzer for large datasets using streaming processing."""
    
    def __init__(self, processor: Optional[StreamingDataProcessor] = None):
        """Initialize large dataset analyzer."""
        self.processor = processor or StreamingDataProcessor()
    
    @require_feature("memory_efficiency")
    def analyze_dataset_statistics(
        self,
        data_source: Union[str, pd.DataFrame, np.ndarray]
    ) -> Dict[str, Any]:
        """Analyze dataset statistics in a memory-efficient way."""
        stats = {
            "total_rows": 0,
            "total_columns": 0,
            "memory_estimate_mb": 0.0,
            "chunk_count": 0,
            "column_stats": {}
        }
        
        first_chunk = True
        running_sums = None
        running_squares = None
        
        for chunk_dataset in self.processor.process_large_dataset(data_source):
            chunk_data = chunk_dataset.data
            
            # Update basic stats
            stats["total_rows"] += chunk_data.shape[0]
            stats["chunk_count"] += 1
            
            if first_chunk:
                stats["total_columns"] = chunk_data.shape[1]
                running_sums = np.zeros(chunk_data.shape[1])
                running_squares = np.zeros(chunk_data.shape[1])
                first_chunk = False
            
            # Update running statistics
            chunk_sums = np.nansum(chunk_data, axis=0)
            chunk_squares = np.nansum(chunk_data ** 2, axis=0)
            
            running_sums += chunk_sums
            running_squares += chunk_squares
        
        # Calculate final statistics
        if stats["total_rows"] > 0:
            means = running_sums / stats["total_rows"]
            variances = (running_squares / stats["total_rows"]) - (means ** 2)
            std_devs = np.sqrt(np.maximum(variances, 0))
            
            stats["column_stats"] = {
                "means": means.tolist(),
                "std_devs": std_devs.tolist(),
                "variances": variances.tolist()
            }
        
        # Estimate memory usage
        stats["memory_estimate_mb"] = (
            stats["total_rows"] * stats["total_columns"] * 8  # 8 bytes per float64
        ) / 1024 / 1024
        
        return stats
    
    @require_feature("memory_efficiency")
    def detect_anomaly_candidates(
        self,
        data_source: Union[str, pd.DataFrame, np.ndarray],
        threshold_factor: float = 3.0
    ) -> Dict[str, Any]:
        """Detect potential anomaly candidates using streaming analysis."""
        candidates = {
            "outlier_indices": [],
            "outlier_scores": [],
            "total_candidates": 0,
            "processing_chunks": 0
        }
        
        # First pass: calculate global statistics
        stats = self.analyze_dataset_statistics(data_source)
        means = np.array(stats["column_stats"]["means"])
        std_devs = np.array(stats["column_stats"]["std_devs"])
        
        # Second pass: identify outliers
        current_idx = 0
        
        for chunk_dataset in self.processor.process_large_dataset(data_source):
            chunk_data = chunk_dataset.data
            candidates["processing_chunks"] += 1
            
            # Calculate z-scores for each point
            z_scores = np.abs((chunk_data - means) / (std_devs + 1e-8))
            max_z_scores = np.max(z_scores, axis=1)
            
            # Find outlier candidates
            outlier_mask = max_z_scores > threshold_factor
            outlier_indices = np.where(outlier_mask)[0] + current_idx
            
            candidates["outlier_indices"].extend(outlier_indices.tolist())
            candidates["outlier_scores"].extend(max_z_scores[outlier_mask].tolist())
            
            current_idx += len(chunk_data)
        
        candidates["total_candidates"] = len(candidates["outlier_indices"])
        
        return candidates


# Utility functions for memory monitoring
def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    import psutil
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def monitor_memory_usage(func):
    """Decorator to monitor memory usage of a function."""
    def wrapper(*args, **kwargs):
        memory_before = get_memory_usage()
        result = func(*args, **kwargs)
        memory_after = get_memory_usage()
        
        print(f"Memory usage: {memory_before:.1f}MB -> {memory_after:.1f}MB "
              f"(+{memory_after - memory_before:.1f}MB)")
        
        return result
    return wrapper