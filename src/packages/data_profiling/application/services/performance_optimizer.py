"""Performance optimization service for data profiling."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Iterator
import logging
import time
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import psutil
import gc
import json

logger = logging.getLogger(__name__)


class SamplingStrategy:
    """Intelligent sampling strategies for large datasets."""
    
    @staticmethod
    def systematic_sampling(df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
        """Systematic sampling with regular intervals."""
        if len(df) <= sample_size:
            return df
        
        interval = len(df) // sample_size
        indices = range(0, len(df), interval)[:sample_size]
        return df.iloc[list(indices)]
    
    @staticmethod
    def stratified_sampling(df: pd.DataFrame, sample_size: int, 
                           stratify_column: Optional[str] = None) -> pd.DataFrame:
        """Stratified sampling to maintain data distribution."""
        if len(df) <= sample_size:
            return df
        
        if stratify_column and stratify_column in df.columns:
            # Sample proportionally from each stratum
            value_counts = df[stratify_column].value_counts()
            proportions = value_counts / len(df)
            
            sampled_dfs = []
            for value, proportion in proportions.items():
                stratum_df = df[df[stratify_column] == value]
                stratum_sample_size = max(1, int(sample_size * proportion))
                
                if len(stratum_df) <= stratum_sample_size:
                    sampled_dfs.append(stratum_df)
                else:
                    sampled_dfs.append(stratum_df.sample(n=stratum_sample_size, random_state=42))
            
            return pd.concat(sampled_dfs, ignore_index=True)
        else:
            # Fallback to random sampling
            return df.sample(n=sample_size, random_state=42)
    
    @staticmethod
    def reservoir_sampling(df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
        """Reservoir sampling for streaming data or very large datasets."""
        if len(df) <= sample_size:
            return df
        
        # Initialize reservoir with first sample_size rows
        reservoir = df.head(sample_size).copy()
        
        # Process remaining rows
        for i in range(sample_size, len(df)):
            # Generate random index
            j = np.random.randint(0, i + 1)
            
            # If j is within reservoir size, replace that element
            if j < sample_size:
                reservoir.iloc[j] = df.iloc[i]
        
        return reservoir
    
    @staticmethod
    def adaptive_sampling(df: pd.DataFrame, target_size_mb: float = 100.0) -> pd.DataFrame:
        """Adaptive sampling based on memory usage."""
        current_size_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        
        if current_size_mb <= target_size_mb:
            return df
        
        # Calculate required sampling ratio
        sampling_ratio = target_size_mb / current_size_mb
        sample_size = max(1000, int(len(df) * sampling_ratio))
        
        return df.sample(n=min(sample_size, len(df)), random_state=42)


class MemoryOptimizer:
    """Memory optimization utilities for data profiling."""
    
    @staticmethod
    def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame data types to reduce memory usage."""
        optimized_df = df.copy()
        
        for col in optimized_df.columns:
            col_type = optimized_df[col].dtype
            
            # Optimize numeric types
            if pd.api.types.is_integer_dtype(col_type):
                c_min = optimized_df[col].min()
                c_max = optimized_df[col].max()
                
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    optimized_df[col] = optimized_df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    optimized_df[col] = optimized_df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    optimized_df[col] = optimized_df[col].astype(np.int32)
                    
            elif pd.api.types.is_float_dtype(col_type):
                c_min = optimized_df[col].min()
                c_max = optimized_df[col].max()
                
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    optimized_df[col] = optimized_df[col].astype(np.float32)
            
            # Convert to categorical if beneficial
            elif pd.api.types.is_object_dtype(col_type):
                num_unique_values = optimized_df[col].nunique()
                num_total_values = len(optimized_df[col])
                
                if num_unique_values / num_total_values < 0.5:  # Less than 50% unique
                    optimized_df[col] = optimized_df[col].astype('category')
        
        return optimized_df
    
    @staticmethod
    def chunk_processor(df: pd.DataFrame, chunk_size: int = 10000) -> Iterator[pd.DataFrame]:
        """Process DataFrame in chunks to manage memory."""
        for i in range(0, len(df), chunk_size):
            yield df.iloc[i:i + chunk_size]
    
    @staticmethod
    def get_memory_usage(df: pd.DataFrame) -> Dict[str, float]:
        """Get detailed memory usage information."""
        memory_usage = df.memory_usage(deep=True)
        
        return {
            'total_mb': memory_usage.sum() / (1024 * 1024),
            'per_column_mb': {col: usage / (1024 * 1024) 
                             for col, usage in memory_usage.items()},
            'per_row_bytes': memory_usage.sum() / len(df) if len(df) > 0 else 0
        }


class ParallelProcessor:
    """Parallel processing utilities for data profiling."""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or min(mp.cpu_count(), 8)
    
    def parallel_column_analysis(self, df: pd.DataFrame, 
                                analysis_func: callable) -> Dict[str, Any]:
        """Analyze columns in parallel."""
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks for each column
            future_to_column = {
                executor.submit(analysis_func, df[col], col): col 
                for col in df.columns
            }
            
            # Collect results
            for future in as_completed(future_to_column):
                column = future_to_column[future]
                try:
                    results[column] = future.result()
                except Exception as e:
                    logger.error(f"Error processing column {column}: {e}")
                    results[column] = {'error': str(e)}
        
        return results
    
    def parallel_table_analysis(self, table_configs: List[Dict[str, Any]], 
                               analysis_func: callable) -> Dict[str, Any]:
        """Analyze multiple tables in parallel."""
        results = {}
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks for each table
            future_to_table = {
                executor.submit(analysis_func, config): config.get('name', f'table_{i}') 
                for i, config in enumerate(table_configs)
            }
            
            # Collect results
            for future in as_completed(future_to_table):
                table_name = future_to_table[future]
                try:
                    results[table_name] = future.result()
                except Exception as e:
                    logger.error(f"Error processing table {table_name}: {e}")
                    results[table_name] = {'error': str(e)}
        
        return results


class CacheManager:
    """Cache management for profiling results."""
    
    def __init__(self, cache_size_mb: float = 500.0):
        self.cache_size_mb = cache_size_mb
        self.cache: Dict[str, Tuple[Any, float, float]] = {}  # {key: (data, timestamp, size_mb)}
        self.total_cache_size = 0.0
    
    def get_cache_key(self, source_path: str, last_modified: float, 
                     profiling_config: Dict[str, Any]) -> str:
        """Generate cache key for profiling results."""
        import hashlib
        
        config_str = json.dumps(profiling_config, sort_keys=True)
        key_string = f"{source_path}:{last_modified}:{config_str}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, cache_key: str) -> Optional[Any]:
        """Get cached result."""
        if cache_key in self.cache:
            data, timestamp, size_mb = self.cache[cache_key]
            # Check if cache entry is still valid (24 hours)
            if time.time() - timestamp < 24 * 3600:
                return data
            else:
                # Remove expired entry
                self.remove(cache_key)
        return None
    
    def put(self, cache_key: str, data: Any, size_mb: float) -> None:
        """Cache profiling result."""
        # Check if we need to evict old entries
        while self.total_cache_size + size_mb > self.cache_size_mb and self.cache:
            self._evict_oldest()
        
        self.cache[cache_key] = (data, time.time(), size_mb)
        self.total_cache_size += size_mb
    
    def remove(self, cache_key: str) -> None:
        """Remove cache entry."""
        if cache_key in self.cache:
            _, _, size_mb = self.cache[cache_key]
            del self.cache[cache_key]
            self.total_cache_size -= size_mb
    
    def _evict_oldest(self) -> None:
        """Evict oldest cache entry."""
        if not self.cache:
            return
        
        oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
        self.remove(oldest_key)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self.total_cache_size = 0.0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'total_entries': len(self.cache),
            'total_size_mb': self.total_cache_size,
            'max_size_mb': self.cache_size_mb,
            'usage_percentage': (self.total_cache_size / self.cache_size_mb) * 100
        }


class PerformanceOptimizer:
    """Main performance optimization service for data profiling."""
    
    def __init__(self, 
                 max_workers: Optional[int] = None,
                 cache_size_mb: float = 500.0,
                 memory_limit_mb: float = 1000.0):
        self.parallel_processor = ParallelProcessor(max_workers)
        self.cache_manager = CacheManager(cache_size_mb)
        self.memory_limit_mb = memory_limit_mb
    
    def optimize_profiling_strategy(self, df: pd.DataFrame, 
                                  config: Dict[str, Any]) -> Dict[str, Any]:
        """Determine optimal profiling strategy based on data characteristics."""
        data_size_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        row_count = len(df)
        column_count = len(df.columns)
        
        # Determine strategy based on data size
        if data_size_mb <= 50:
            strategy = "full_analysis"
            sampling_needed = False
        elif data_size_mb <= 200:
            strategy = "optimized_analysis"
            sampling_needed = False
        elif data_size_mb <= 1000:
            strategy = "sampled_analysis"
            sampling_needed = True
        else:
            strategy = "chunked_analysis"
            sampling_needed = True
        
        # Determine sampling configuration
        sampling_config = {}
        if sampling_needed:
            if data_size_mb > self.memory_limit_mb:
                target_size_mb = self.memory_limit_mb * 0.8  # 80% of limit
                sampling_config = {
                    'method': 'adaptive',
                    'target_size_mb': target_size_mb
                }
            else:
                sampling_config = {
                    'method': 'systematic',
                    'sample_size': min(100000, row_count // 10)
                }
        
        # Determine parallelization strategy
        parallel_config = {
            'column_parallel': column_count > 20,
            'chunk_parallel': data_size_mb > 500,
            'max_workers': min(self.parallel_processor.max_workers, column_count)
        }
        
        return {
            'strategy': strategy,
            'data_size_mb': data_size_mb,
            'sampling_config': sampling_config,
            'parallel_config': parallel_config,
            'memory_optimization': data_size_mb > 100,
            'caching_enabled': True
        }
    
    def apply_sampling(self, df: pd.DataFrame, 
                      sampling_config: Dict[str, Any]) -> pd.DataFrame:
        """Apply intelligent sampling to the dataset."""
        if not sampling_config:
            return df
        
        method = sampling_config.get('method', 'systematic')
        
        if method == 'systematic':
            sample_size = sampling_config.get('sample_size', 10000)
            return SamplingStrategy.systematic_sampling(df, sample_size)
        
        elif method == 'stratified':
            sample_size = sampling_config.get('sample_size', 10000)
            stratify_column = sampling_config.get('stratify_column')
            return SamplingStrategy.stratified_sampling(df, sample_size, stratify_column)
        
        elif method == 'reservoir':
            sample_size = sampling_config.get('sample_size', 10000)
            return SamplingStrategy.reservoir_sampling(df, sample_size)
        
        elif method == 'adaptive':
            target_size_mb = sampling_config.get('target_size_mb', 100.0)
            return SamplingStrategy.adaptive_sampling(df, target_size_mb)
        
        else:
            logger.warning(f"Unknown sampling method: {method}")
            return df
    
    def optimize_memory_usage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage."""
        initial_memory = df.memory_usage(deep=True).sum() / (1024 * 1024)
        
        # Optimize data types
        optimized_df = MemoryOptimizer.optimize_dtypes(df)
        
        # Force garbage collection
        gc.collect()
        
        final_memory = optimized_df.memory_usage(deep=True).sum() / (1024 * 1024)
        
        logger.info(f"Memory optimization: {initial_memory:.2f}MB -> {final_memory:.2f}MB "
                   f"({((initial_memory - final_memory) / initial_memory * 100):.1f}% reduction)")
        
        return optimized_df
    
    def check_system_resources(self) -> Dict[str, Any]:
        """Check available system resources."""
        memory = psutil.virtual_memory()
        cpu_count = psutil.cpu_count()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        return {
            'memory': {
                'total_gb': memory.total / (1024 ** 3),
                'available_gb': memory.available / (1024 ** 3),
                'used_percent': memory.percent
            },
            'cpu': {
                'count': cpu_count,
                'usage_percent': cpu_percent
            },
            'recommendations': self._generate_resource_recommendations(memory, cpu_percent)
        }
    
    def _generate_resource_recommendations(self, memory, cpu_percent: float) -> List[str]:
        """Generate resource optimization recommendations."""
        recommendations = []
        
        if memory.percent > 85:
            recommendations.append("High memory usage detected. Consider reducing dataset size or using sampling.")
        
        if cpu_percent > 90:
            recommendations.append("High CPU usage detected. Consider reducing parallelization.")
        
        if memory.available / (1024 ** 3) < 2:  # Less than 2GB available
            recommendations.append("Low available memory. Enable aggressive memory optimization.")
        
        return recommendations
    
    def profile_with_optimization(self, df: pd.DataFrame, 
                                 profiling_func: callable,
                                 config: Dict[str, Any]) -> Dict[str, Any]:
        """Profile dataset with automatic performance optimization."""
        start_time = time.time()
        
        # Check system resources
        system_resources = self.check_system_resources()
        
        # Determine optimization strategy
        optimization_strategy = self.optimize_profiling_strategy(df, config)
        
        # Apply memory optimization if needed
        if optimization_strategy['memory_optimization']:
            df = self.optimize_memory_usage(df)
        
        # Apply sampling if needed
        if optimization_strategy['sampling_config']:
            original_size = len(df)
            df = self.apply_sampling(df, optimization_strategy['sampling_config'])
            logger.info(f"Applied sampling: {original_size} -> {len(df)} rows")
        
        # Execute profiling with performance monitoring
        try:
            if optimization_strategy['parallel_config']['column_parallel']:
                # Parallel column analysis
                logger.info("Using parallel column analysis")
                profiling_results = self.parallel_processor.parallel_column_analysis(df, profiling_func)
            else:
                # Standard sequential analysis
                profiling_results = profiling_func(df)
            
            execution_time = time.time() - start_time
            
            return {
                'profiling_results': profiling_results,
                'optimization_strategy': optimization_strategy,
                'system_resources': system_resources,
                'performance_metrics': {
                    'execution_time_seconds': execution_time,
                    'final_dataset_size_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
                    'rows_processed': len(df),
                    'columns_processed': len(df.columns)
                },
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Profiling failed: {e}")
            return {
                'error': str(e),
                'optimization_strategy': optimization_strategy,
                'system_resources': system_resources,
                'success': False
            }


class IncrementalProfiler:
    """Incremental profiling for large datasets and streaming data."""
    
    def __init__(self, chunk_size: int = 10000):
        self.chunk_size = chunk_size
        self.aggregated_stats = {}
        self.processed_chunks = 0
        self.total_rows = 0
    
    def process_chunk(self, chunk: pd.DataFrame) -> Dict[str, Any]:
        """Process a single chunk and update aggregated statistics."""
        chunk_stats = {}
        
        for column in chunk.columns:
            series = chunk[column]
            
            # Calculate chunk statistics
            if pd.api.types.is_numeric_dtype(series):
                chunk_stats[column] = {
                    'count': len(series.dropna()),
                    'sum': series.sum(),
                    'sum_sq': (series ** 2).sum(),
                    'min': series.min(),
                    'max': series.max(),
                    'null_count': series.isnull().sum()
                }
            else:
                chunk_stats[column] = {
                    'count': len(series.dropna()),
                    'unique_values': set(series.dropna().unique()),
                    'null_count': series.isnull().sum()
                }
        
        # Update aggregated statistics
        self._update_aggregated_stats(chunk_stats)
        self.processed_chunks += 1
        self.total_rows += len(chunk)
        
        return chunk_stats
    
    def _update_aggregated_stats(self, chunk_stats: Dict[str, Any]) -> None:
        """Update aggregated statistics with chunk statistics."""
        for column, stats in chunk_stats.items():
            if column not in self.aggregated_stats:
                self.aggregated_stats[column] = stats.copy()
                if 'unique_values' in stats:
                    self.aggregated_stats[column]['unique_values'] = set(stats['unique_values'])
            else:
                agg_stats = self.aggregated_stats[column]
                
                if 'sum' in stats:  # Numeric column
                    agg_stats['count'] += stats['count']
                    agg_stats['sum'] += stats['sum']
                    agg_stats['sum_sq'] += stats['sum_sq']
                    agg_stats['min'] = min(agg_stats['min'], stats['min'])
                    agg_stats['max'] = max(agg_stats['max'], stats['max'])
                    agg_stats['null_count'] += stats['null_count']
                else:  # Categorical column
                    agg_stats['count'] += stats['count']
                    agg_stats['unique_values'].update(stats['unique_values'])
                    agg_stats['null_count'] += stats['null_count']
    
    def get_final_statistics(self) -> Dict[str, Any]:
        """Calculate final statistics from aggregated data."""
        final_stats = {}
        
        for column, agg_stats in self.aggregated_stats.items():
            if 'sum' in agg_stats:  # Numeric column
                count = agg_stats['count']
                if count > 0:
                    mean = agg_stats['sum'] / count
                    variance = (agg_stats['sum_sq'] / count) - (mean ** 2)
                    std_dev = variance ** 0.5 if variance >= 0 else 0
                else:
                    mean = std_dev = 0
                
                final_stats[column] = {
                    'count': count,
                    'mean': mean,
                    'std_dev': std_dev,
                    'min': agg_stats['min'],
                    'max': agg_stats['max'],
                    'null_count': agg_stats['null_count'],
                    'completeness_ratio': count / self.total_rows if self.total_rows > 0 else 0
                }
            else:  # Categorical column
                final_stats[column] = {
                    'count': agg_stats['count'],
                    'unique_count': len(agg_stats['unique_values']),
                    'null_count': agg_stats['null_count'],
                    'completeness_ratio': agg_stats['count'] / self.total_rows if self.total_rows > 0 else 0
                }
        
        return {
            'column_statistics': final_stats,
            'dataset_statistics': {
                'total_rows': self.total_rows,
                'total_chunks': self.processed_chunks,
                'average_chunk_size': self.total_rows / self.processed_chunks if self.processed_chunks > 0 else 0
            }
        }