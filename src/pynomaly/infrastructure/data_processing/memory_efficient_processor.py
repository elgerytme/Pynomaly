"""Memory-efficient data processing for large datasets.

This module provides streaming and memory-optimized data processing capabilities
for handling large datasets that don't fit in memory, while maintaining clean
architecture principles.
"""

from __future__ import annotations

import gc
import psutil
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional, Protocol, Union
from datetime import datetime

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from pynomaly.domain.entities import Dataset
from pynomaly.domain.exceptions import DomainError, InfrastructureError
from pynomaly.infrastructure.monitoring import get_monitor, monitor_operation


@dataclass
class MemoryMetrics:
    """Memory usage metrics."""
    total_mb: float
    available_mb: float
    used_mb: float
    percent_used: float
    process_mb: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_mb': self.total_mb,
            'available_mb': self.available_mb,
            'used_mb': self.used_mb,
            'percent_used': self.percent_used,
            'process_mb': self.process_mb,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class DataChunk:
    """Container for data chunks in streaming processing."""
    data: pd.DataFrame
    chunk_id: int
    total_chunks: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def size_mb(self) -> float:
        """Get chunk size in MB."""
        return self.data.memory_usage(deep=True).sum() / 1024 / 1024
    
    @property
    def shape(self) -> tuple:
        """Get data shape."""
        return self.data.shape


class DataProcessorProtocol(Protocol):
    """Protocol for data processors."""
    
    def process_chunk(self, chunk: DataChunk) -> DataChunk:
        """Process a single data chunk."""
        ...
    
    def finalize(self) -> Any:
        """Finalize processing and return results."""
        ...


def get_memory_usage() -> MemoryMetrics:
    """Get current memory usage metrics."""
    process = psutil.Process()
    memory = psutil.virtual_memory()
    
    return MemoryMetrics(
        total_mb=memory.total / 1024 / 1024,
        available_mb=memory.available / 1024 / 1024,
        used_mb=memory.used / 1024 / 1024,
        percent_used=memory.percent,
        process_mb=process.memory_info().rss / 1024 / 1024
    )


@contextmanager
def monitor_memory_usage(operation_name: str = "data_processing"):
    """Context manager to monitor memory usage during operations."""
    monitor = get_monitor()
    
    start_metrics = get_memory_usage()
    gc.collect()  # Clean up before starting
    
    with monitor_operation(operation_name, "data_processing") as request_id:
        try:
            yield start_metrics
        finally:
            end_metrics = get_memory_usage()
            
            # Log memory usage
            monitor.info(
                f"Memory usage for {operation_name}",
                operation=operation_name,
                component="memory_monitor",
                request_id=request_id,
                start_memory_mb=start_metrics.process_mb,
                end_memory_mb=end_metrics.process_mb,
                memory_delta_mb=end_metrics.process_mb - start_metrics.process_mb,
                peak_memory_percent=end_metrics.percent_used
            )


class MemoryOptimizedDataLoader:
    """Memory-efficient data loader with chunking capabilities."""
    
    def __init__(
        self,
        chunk_size: int = 10000,
        memory_limit_mb: float = 1000.0,
        auto_optimize: bool = True
    ):
        """Initialize memory-optimized data loader.
        
        Args:
            chunk_size: Number of rows per chunk
            memory_limit_mb: Memory limit in MB
            auto_optimize: Automatically optimize chunk size based on memory
        """
        self.chunk_size = chunk_size
        self.memory_limit_mb = memory_limit_mb
        self.auto_optimize = auto_optimize
        self._optimal_chunk_size: Optional[int] = None
        
    def load_csv(
        self,
        file_path: Union[str, Path],
        **pandas_kwargs
    ) -> Iterator[DataChunk]:
        """Load CSV file in chunks.
        
        Args:
            file_path: Path to CSV file
            **pandas_kwargs: Additional arguments for pandas.read_csv
            
        Yields:
            DataChunk objects
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise InfrastructureError(f"File not found: {file_path}")
        
        with monitor_memory_usage(f"load_csv_{file_path.name}"):
            # Determine optimal chunk size if needed
            if self.auto_optimize and self._optimal_chunk_size is None:
                self._optimal_chunk_size = self._determine_optimal_chunk_size(file_path)
                chunk_size = self._optimal_chunk_size
            else:
                chunk_size = self.chunk_size
            
            # Read in chunks
            chunk_id = 0
            total_chunks = None
            
            for chunk_df in pd.read_csv(file_path, chunksize=chunk_size, **pandas_kwargs):
                # Optimize data types to save memory
                chunk_df = self._optimize_datatypes(chunk_df)
                
                yield DataChunk(
                    data=chunk_df,
                    chunk_id=chunk_id,
                    total_chunks=total_chunks,
                    metadata={
                        'source_file': str(file_path),
                        'chunk_size': chunk_size,
                        'memory_optimized': True
                    }
                )
                chunk_id += 1
                
                # Memory check
                self._check_memory_usage()
    
    def load_parquet(
        self,
        file_path: Union[str, Path],
        **pandas_kwargs
    ) -> Iterator[DataChunk]:
        """Load Parquet file in chunks.
        
        Args:
            file_path: Path to Parquet file
            **pandas_kwargs: Additional arguments for pandas.read_parquet
            
        Yields:
            DataChunk objects
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise InfrastructureError(f"File not found: {file_path}")
        
        with monitor_memory_usage(f"load_parquet_{file_path.name}"):
            # For Parquet, we need to determine the number of rows first
            # and then read in chunks using row groups or manual chunking
            try:
                # Read metadata to determine size
                import pyarrow.parquet as pq
                parquet_file = pq.ParquetFile(file_path)
                total_rows = parquet_file.metadata.num_rows
                
                chunk_size = self._optimal_chunk_size or self.chunk_size
                total_chunks = (total_rows + chunk_size - 1) // chunk_size
                
                chunk_id = 0
                for start_row in range(0, total_rows, chunk_size):
                    end_row = min(start_row + chunk_size, total_rows)
                    
                    # Read specific row range
                    chunk_df = pd.read_parquet(
                        file_path, 
                        **pandas_kwargs
                    ).iloc[start_row:end_row]
                    
                    # Optimize data types
                    chunk_df = self._optimize_datatypes(chunk_df)
                    
                    yield DataChunk(
                        data=chunk_df,
                        chunk_id=chunk_id,
                        total_chunks=total_chunks,
                        metadata={
                            'source_file': str(file_path),
                            'chunk_size': chunk_size,
                            'row_range': (start_row, end_row),
                            'memory_optimized': True
                        }
                    )
                    chunk_id += 1
                    self._check_memory_usage()
                    
            except ImportError:
                # Fallback to pandas chunking if pyarrow not available
                warnings.warn("PyArrow not available, using pandas chunking")
                df = pd.read_parquet(file_path, **pandas_kwargs)
                yield from self._chunk_dataframe(df, str(file_path))
    
    def load_dataset(self, dataset: Dataset) -> Iterator[DataChunk]:
        """Load dataset in chunks.
        
        Args:
            dataset: Dataset entity
            
        Yields:
            DataChunk objects
        """
        if dataset.data is not None:
            # Dataset already loaded in memory, chunk it
            yield from self._chunk_dataframe(dataset.data, f"dataset_{dataset.id}")
        elif hasattr(dataset, 'file_path') and dataset.file_path:
            # Load from file
            file_path = Path(dataset.file_path)
            if file_path.suffix.lower() == '.csv':
                yield from self.load_csv(file_path)
            elif file_path.suffix.lower() in ['.parquet', '.pq']:
                yield from self.load_parquet(file_path)
            else:
                raise InfrastructureError(f"Unsupported file format: {file_path.suffix}")
        else:
            raise InfrastructureError("Dataset has no data or file path")
    
    def _chunk_dataframe(self, df: pd.DataFrame, source: str) -> Iterator[DataChunk]:
        """Chunk an existing DataFrame."""
        chunk_size = self._optimal_chunk_size or self.chunk_size
        total_chunks = (len(df) + chunk_size - 1) // chunk_size
        
        for chunk_id, start_idx in enumerate(range(0, len(df), chunk_size)):
            end_idx = min(start_idx + chunk_size, len(df))
            chunk_df = df.iloc[start_idx:end_idx].copy()
            
            # Optimize data types
            chunk_df = self._optimize_datatypes(chunk_df)
            
            yield DataChunk(
                data=chunk_df,
                chunk_id=chunk_id,
                total_chunks=total_chunks,
                metadata={
                    'source': source,
                    'chunk_size': chunk_size,
                    'row_range': (start_idx, end_idx),
                    'memory_optimized': True
                }
            )
            self._check_memory_usage()
    
    def _determine_optimal_chunk_size(self, file_path: Path) -> int:
        """Determine optimal chunk size based on available memory and file size."""
        try:
            # Sample a small chunk to estimate memory usage per row
            sample_df = pd.read_csv(file_path, nrows=1000)
            sample_df = self._optimize_datatypes(sample_df)
            
            # Calculate memory per row
            memory_per_row = sample_df.memory_usage(deep=True).sum() / len(sample_df)
            
            # Available memory for processing (leave some buffer)
            available_memory = get_memory_usage().available_mb * 0.8 * 1024 * 1024  # bytes
            target_memory = min(available_memory, self.memory_limit_mb * 1024 * 1024)
            
            # Calculate optimal chunk size
            optimal_size = int(target_memory / memory_per_row)
            
            # Ensure reasonable bounds
            optimal_size = max(1000, min(optimal_size, 100000))
            
            get_monitor().info(
                f"Determined optimal chunk size: {optimal_size}",
                component="memory_optimizer",
                operation="chunk_size_optimization",
                memory_per_row_bytes=memory_per_row,
                target_memory_mb=target_memory / 1024 / 1024,
                optimal_chunk_size=optimal_size
            )
            
            return optimal_size
            
        except Exception as e:
            get_monitor().warning(
                f"Could not determine optimal chunk size: {e}",
                component="memory_optimizer",
                operation="chunk_size_optimization"
            )
            return self.chunk_size
    
    def _optimize_datatypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame data types to reduce memory usage."""
        original_memory = df.memory_usage(deep=True).sum()
        
        for col in df.columns:
            col_type = df[col].dtype
            
            # Optimize numeric types
            if is_numeric_dtype(df[col]):
                if col_type == 'int64':
                    # Try smaller integer types
                    if df[col].min() >= -128 and df[col].max() <= 127:
                        df[col] = df[col].astype('int8')
                    elif df[col].min() >= -32768 and df[col].max() <= 32767:
                        df[col] = df[col].astype('int16')
                    elif df[col].min() >= -2147483648 and df[col].max() <= 2147483647:
                        df[col] = df[col].astype('int32')
                
                elif col_type == 'float64':
                    # Try float32 if precision allows
                    if df[col].isna().sum() == 0:  # No NaN values
                        df[col] = pd.to_numeric(df[col], downcast='float')
            
            # Optimize object/string types
            elif col_type == 'object':
                # Try categorical if low cardinality
                unique_count = df[col].nunique()
                total_count = len(df[col])
                
                if unique_count / total_count < 0.5 and unique_count < 1000:
                    df[col] = df[col].astype('category')
        
        optimized_memory = df.memory_usage(deep=True).sum()
        reduction = (original_memory - optimized_memory) / original_memory * 100
        
        if reduction > 5:  # Log significant reductions
            get_monitor().info(
                f"Memory optimization reduced usage by {reduction:.1f}%",
                component="memory_optimizer",
                operation="datatype_optimization",
                original_memory_mb=original_memory / 1024 / 1024,
                optimized_memory_mb=optimized_memory / 1024 / 1024,
                reduction_percent=reduction
            )
        
        return df
    
    def _check_memory_usage(self) -> None:
        """Check memory usage and warn if approaching limits."""
        metrics = get_memory_usage()
        
        if metrics.percent_used > 90:
            get_monitor().warning(
                f"High memory usage: {metrics.percent_used:.1f}%",
                component="memory_monitor",
                operation="memory_check",
                memory_percent=metrics.percent_used,
                available_mb=metrics.available_mb
            )
            
            # Force garbage collection
            gc.collect()
        
        elif metrics.process_mb > self.memory_limit_mb:
            get_monitor().warning(
                f"Process memory ({metrics.process_mb:.1f}MB) exceeds limit ({self.memory_limit_mb}MB)",
                component="memory_monitor",
                operation="memory_check",
                process_memory_mb=metrics.process_mb,
                memory_limit_mb=self.memory_limit_mb
            )


class StreamingDataProcessor:
    """Streaming data processor for large datasets."""
    
    def __init__(
        self,
        loader: Optional[MemoryOptimizedDataLoader] = None,
        monitor_memory: bool = True
    ):
        """Initialize streaming processor.
        
        Args:
            loader: Data loader instance
            monitor_memory: Enable memory monitoring
        """
        self.loader = loader or MemoryOptimizedDataLoader()
        self.monitor_memory = monitor_memory
        self._processed_chunks = 0
        self._total_rows = 0
        
    def process_dataset(
        self,
        dataset: Dataset,
        processor: DataProcessorProtocol,
        progress_callback: Optional[callable] = None
    ) -> Any:
        """Process dataset using streaming approach.
        
        Args:
            dataset: Dataset to process
            processor: Processor implementing DataProcessorProtocol
            progress_callback: Optional callback for progress updates
            
        Returns:
            Processing results
        """
        operation_name = f"stream_process_{dataset.id}"
        
        with monitor_memory_usage(operation_name) if self.monitor_memory else contextmanager(lambda: (yield))():
            self._processed_chunks = 0
            self._total_rows = 0
            
            for chunk in self.loader.load_dataset(dataset):
                # Process chunk
                processed_chunk = processor.process_chunk(chunk)
                
                # Update statistics
                self._processed_chunks += 1
                self._total_rows += len(chunk.data)
                
                # Progress callback
                if progress_callback:
                    progress_callback(
                        chunk_id=chunk.chunk_id,
                        total_chunks=chunk.total_chunks,
                        rows_processed=self._total_rows,
                        chunk_size=len(chunk.data)
                    )
                
                # Log progress
                if self._processed_chunks % 10 == 0:  # Log every 10 chunks
                    get_monitor().info(
                        f"Processed {self._processed_chunks} chunks, {self._total_rows} rows",
                        operation=operation_name,
                        component="streaming_processor",
                        chunks_processed=self._processed_chunks,
                        total_rows=self._total_rows,
                        current_chunk_size=len(chunk.data)
                    )
            
            # Finalize processing
            result = processor.finalize()
            
            get_monitor().info(
                f"Streaming processing complete",
                operation=operation_name,
                component="streaming_processor",
                total_chunks=self._processed_chunks,
                total_rows=self._total_rows
            )
            
            return result
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            'processed_chunks': self._processed_chunks,
            'total_rows': self._total_rows,
            'memory_usage': get_memory_usage().to_dict()
        }


class LargeDatasetAnalyzer:
    """Analyzer for large datasets using streaming approach."""
    
    def __init__(self, loader: Optional[MemoryOptimizedDataLoader] = None):
        """Initialize analyzer."""
        self.loader = loader or MemoryOptimizedDataLoader()
        
    def analyze_dataset(self, dataset: Dataset) -> Dict[str, Any]:
        """Analyze dataset statistics using streaming.
        
        Args:
            dataset: Dataset to analyze
            
        Returns:
            Dataset statistics
        """
        with monitor_memory_usage("dataset_analysis"):
            stats = {
                'total_rows': 0,
                'total_columns': 0,
                'column_info': {},
                'memory_usage_mb': 0,
                'chunks_processed': 0,
                'numeric_columns': [],
                'categorical_columns': [],
                'missing_values': {}
            }
            
            first_chunk = True
            
            for chunk in self.loader.load_dataset(dataset):
                chunk_data = chunk.data
                
                # Initialize on first chunk
                if first_chunk:
                    stats['total_columns'] = len(chunk_data.columns)
                    stats['column_info'] = {col: {'dtype': str(chunk_data[col].dtype)} 
                                          for col in chunk_data.columns}
                    
                    # Identify column types
                    for col in chunk_data.columns:
                        if is_numeric_dtype(chunk_data[col]):
                            stats['numeric_columns'].append(col)
                        else:
                            stats['categorical_columns'].append(col)
                        
                        stats['missing_values'][col] = 0
                    
                    first_chunk = False
                
                # Update statistics
                stats['total_rows'] += len(chunk_data)
                stats['memory_usage_mb'] += chunk.size_mb
                stats['chunks_processed'] += 1
                
                # Update missing values count
                for col in chunk_data.columns:
                    stats['missing_values'][col] += chunk_data[col].isna().sum()
                
                # Update column statistics
                for col in stats['numeric_columns']:
                    if col in chunk_data.columns:
                        col_stats = stats['column_info'][col]
                        
                        # Initialize or update min/max
                        col_min = chunk_data[col].min()
                        col_max = chunk_data[col].max()
                        
                        if 'min' not in col_stats or col_min < col_stats['min']:
                            col_stats['min'] = col_min
                        if 'max' not in col_stats or col_max > col_stats['max']:
                            col_stats['max'] = col_max
            
            # Calculate missing value percentages
            for col in stats['missing_values']:
                count = stats['missing_values'][col]
                stats['missing_values'][col] = {
                    'count': count,
                    'percentage': (count / stats['total_rows'] * 100) if stats['total_rows'] > 0 else 0
                }
            
            get_monitor().info(
                "Dataset analysis complete",
                operation="dataset_analysis",
                component="dataset_analyzer",
                total_rows=stats['total_rows'],
                total_columns=stats['total_columns'],
                chunks_processed=stats['chunks_processed'],
                memory_usage_mb=stats['memory_usage_mb']
            )
            
            return stats