import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
import time
from functools import lru_cache
import gc

logger = logging.getLogger(__name__)


class PerformanceOptimizer:
    """Service for optimizing data profiling performance through intelligent sampling, 
    parallel processing, and memory management."""
    
    def __init__(self):
        self.cpu_count = multiprocessing.cpu_count()
        self.memory_threshold_mb = 1024  # 1GB threshold
        self.cache_size = 100
    
    def apply_intelligent_sampling(self, df: pd.DataFrame, 
                                  target_size: int = 10000,
                                  target_percentage: Optional[float] = None,
                                  stratify_column: Optional[str] = None) -> pd.DataFrame:
        """Apply intelligent sampling strategies for large datasets."""
        try:
            original_size = len(df)
            
            # No sampling needed if data is already small
            if original_size <= target_size:
                logger.info(f"No sampling needed: {original_size} <= {target_size}")
                return df
            
            # Calculate sample size
            if target_percentage:
                sample_size = int(original_size * target_percentage / 100)
            else:
                sample_size = min(target_size, original_size)
            
            logger.info(f"Applying intelligent sampling: {original_size} -> {sample_size} rows")
            
            # Choose sampling strategy
            if stratify_column and stratify_column in df.columns:
                sampled_df = self._stratified_sampling(df, sample_size, stratify_column)
            else:
                sampled_df = self._adaptive_sampling(df, sample_size)
            
            return sampled_df
            
        except Exception as e:
            logger.error(f"Intelligent sampling failed: {e}")
            # Fallback to simple random sampling
            return df.sample(n=min(target_size, len(df)), random_state=42)
    
    def _stratified_sampling(self, df: pd.DataFrame, sample_size: int, stratify_column: str) -> pd.DataFrame:
        """Perform stratified sampling to maintain distribution."""
        try:
            # Get value counts for stratification
            value_counts = df[stratify_column].value_counts()
            
            # Calculate sample size per stratum
            samples_per_stratum = {}
            for value, count in value_counts.items():
                proportion = count / len(df)
                stratum_sample_size = max(1, int(sample_size * proportion))
                samples_per_stratum[value] = min(stratum_sample_size, count)
            
            # Sample from each stratum
            sampled_dfs = []
            for value, stratum_size in samples_per_stratum.items():
                stratum_df = df[df[stratify_column] == value]
                if len(stratum_df) > 0:
                    sampled_stratum = stratum_df.sample(n=stratum_size, random_state=42)
                    sampled_dfs.append(sampled_stratum)
            
            # Combine all strata
            return pd.concat(sampled_dfs, ignore_index=True)
            
        except Exception as e:
            logger.error(f"Stratified sampling failed: {e}")
            return df.sample(n=sample_size, random_state=42)
    
    def _adaptive_sampling(self, df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
        """Perform adaptive sampling based on data characteristics."""
        try:
            # For time series data, prefer systematic sampling
            if self._is_time_series_data(df):
                return self._systematic_sampling(df, sample_size)
            
            # For data with high variance, prefer cluster sampling
            if self._has_high_variance(df):
                return self._cluster_sampling(df, sample_size)
            
            # Default to random sampling
            return df.sample(n=sample_size, random_state=42)
            
        except Exception as e:
            logger.error(f"Adaptive sampling failed: {e}")
            return df.sample(n=sample_size, random_state=42)
    
    def _systematic_sampling(self, df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
        """Systematic sampling for time series data."""
        try:
            interval = len(df) // sample_size
            indices = list(range(0, len(df), interval))[:sample_size]
            return df.iloc[indices]
            
        except Exception as e:
            logger.error(f"Systematic sampling failed: {e}")
            return df.sample(n=sample_size, random_state=42)
    
    def _cluster_sampling(self, df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
        """Cluster sampling for high-variance data."""
        try:
            # Simple cluster sampling - divide data into chunks and sample from each
            n_clusters = min(10, sample_size // 10)
            cluster_size = len(df) // n_clusters
            
            sampled_dfs = []
            for i in range(n_clusters):
                start_idx = i * cluster_size
                end_idx = (i + 1) * cluster_size if i < n_clusters - 1 else len(df)
                cluster_df = df.iloc[start_idx:end_idx]
                
                if len(cluster_df) > 0:
                    cluster_sample_size = sample_size // n_clusters
                    sampled_cluster = cluster_df.sample(
                        n=min(cluster_sample_size, len(cluster_df)), 
                        random_state=42
                    )
                    sampled_dfs.append(sampled_cluster)
            
            return pd.concat(sampled_dfs, ignore_index=True)
            
        except Exception as e:
            logger.error(f"Cluster sampling failed: {e}")
            return df.sample(n=sample_size, random_state=42)
    
    def _is_time_series_data(self, df: pd.DataFrame) -> bool:
        """Check if data appears to be time series."""
        # Look for datetime columns
        datetime_columns = df.select_dtypes(include=['datetime64']).columns
        return len(datetime_columns) > 0
    
    def _has_high_variance(self, df: pd.DataFrame) -> bool:
        """Check if data has high variance."""
        try:
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) == 0:
                return False
            
            # Calculate coefficient of variation for numeric columns
            cv_scores = []
            for col in numeric_columns:
                mean_val = df[col].mean()
                std_val = df[col].std()
                if mean_val != 0:
                    cv_scores.append(std_val / abs(mean_val))
            
            # High variance if average CV > 0.5
            return np.mean(cv_scores) > 0.5 if cv_scores else False
            
        except Exception:
            return False
    
    def optimize_memory_usage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize memory usage by downcasting numeric types."""
        try:
            logger.info("Optimizing memory usage...")
            
            # Store original memory usage
            original_memory = df.memory_usage(deep=True).sum()
            
            # Create a copy to avoid modifying original
            optimized_df = df.copy()
            
            # Optimize numeric columns
            for col in optimized_df.select_dtypes(include=[np.number]).columns:
                col_type = optimized_df[col].dtype
                
                if col_type != 'object':
                    # Try to downcast
                    if 'int' in str(col_type):
                        optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='integer')
                    elif 'float' in str(col_type):
                        optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='float')
            
            # Optimize string columns
            for col in optimized_df.select_dtypes(include=['object']).columns:
                # Convert to category if cardinality is low
                unique_count = optimized_df[col].nunique()
                total_count = len(optimized_df[col])
                
                if unique_count / total_count < 0.5:  # Less than 50% unique values
                    optimized_df[col] = optimized_df[col].astype('category')
            
            # Calculate memory savings
            new_memory = optimized_df.memory_usage(deep=True).sum()
            memory_savings = ((original_memory - new_memory) / original_memory) * 100
            
            logger.info(f"Memory optimization: {memory_savings:.1f}% reduction")
            
            return optimized_df
            
        except Exception as e:
            logger.error(f"Memory optimization failed: {e}")
            return df
    
    def optimize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types for better performance."""
        try:
            logger.info("Optimizing data types...")
            
            optimized_df = df.copy()
            
            # Optimize object columns
            for col in optimized_df.select_dtypes(include=['object']).columns:
                # Try to convert to datetime
                if self._is_datetime_column(optimized_df[col]):
                    try:
                        optimized_df[col] = pd.to_datetime(optimized_df[col], errors='coerce')
                        continue
                    except:
                        pass
                
                # Try to convert to numeric
                if self._is_numeric_column(optimized_df[col]):
                    try:
                        optimized_df[col] = pd.to_numeric(optimized_df[col], errors='coerce')
                        continue
                    except:
                        pass
                
                # Convert to category if appropriate
                if self._should_be_category(optimized_df[col]):
                    optimized_df[col] = optimized_df[col].astype('category')
            
            return optimized_df
            
        except Exception as e:
            logger.error(f"Data type optimization failed: {e}")
            return df
    
    def _is_datetime_column(self, series: pd.Series) -> bool:
        """Check if a series should be datetime."""
        sample = series.dropna().head(100)
        if len(sample) == 0:
            return False
        
        # Check for common datetime patterns
        datetime_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}',  # YYYY-MM-DD HH:MM:SS
        ]
        
        for pattern in datetime_patterns:
            if sample.astype(str).str.contains(pattern, regex=True).any():
                return True
        
        return False
    
    def _is_numeric_column(self, series: pd.Series) -> bool:
        """Check if a series should be numeric."""
        sample = series.dropna().head(100)
        if len(sample) == 0:
            return False
        
        # Try to convert to numeric
        try:
            pd.to_numeric(sample, errors='raise')
            return True
        except:
            return False
    
    def _should_be_category(self, series: pd.Series) -> bool:
        """Check if a series should be categorical."""
        unique_count = series.nunique()
        total_count = len(series)
        
        # Convert to category if less than 50% unique values and more than 2 unique values
        return unique_count / total_count < 0.5 and unique_count > 2
    
    def parallelize_operation(self, df: pd.DataFrame, 
                            operation: callable,
                            n_partitions: Optional[int] = None) -> pd.DataFrame:
        """Parallelize operations across DataFrame partitions."""
        try:
            if n_partitions is None:
                n_partitions = min(self.cpu_count, 4)
            
            logger.info(f"Parallelizing operation across {n_partitions} partitions")
            
            # Split DataFrame into partitions
            partitions = np.array_split(df, n_partitions)
            
            # Process partitions in parallel
            with ProcessPoolExecutor(max_workers=n_partitions) as executor:
                results = list(executor.map(operation, partitions))
            
            # Combine results
            return pd.concat(results, ignore_index=True)
            
        except Exception as e:
            logger.error(f"Parallel operation failed: {e}")
            # Fallback to sequential processing
            return operation(df)
    
    def process_in_chunks(self, df: pd.DataFrame, 
                         operation: callable,
                         chunk_size: int = 1000) -> pd.DataFrame:
        """Process DataFrame in chunks to manage memory."""
        try:
            logger.info(f"Processing DataFrame in chunks of {chunk_size}")
            
            results = []
            for i in range(0, len(df), chunk_size):
                chunk = df.iloc[i:i + chunk_size]
                result = operation(chunk)
                results.append(result)
                
                # Force garbage collection
                if i % (chunk_size * 10) == 0:
                    gc.collect()
            
            return pd.concat(results, ignore_index=True)
            
        except Exception as e:
            logger.error(f"Chunk processing failed: {e}")
            return operation(df)
    
    @lru_cache(maxsize=100)
    def get_dataset_statistics(self, df_hash: str, df_shape: Tuple[int, int]) -> Dict[str, Any]:
        """Get cached dataset statistics."""
        # This is a placeholder for cached statistics
        return {
            'shape': df_shape,
            'memory_usage_mb': df_shape[0] * df_shape[1] * 8 / (1024 * 1024),  # Rough estimate
            'optimization_recommended': df_shape[0] * df_shape[1] > 1000000
        }
    
    def monitor_performance(self, operation: callable, *args, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Monitor performance of an operation."""
        import psutil
        import os
        
        try:
            # Get initial metrics
            process = psutil.Process(os.getpid())
            start_time = time.time()
            start_memory = process.memory_info().rss / 1024 / 1024  # MB
            start_cpu = process.cpu_percent()
            
            # Execute operation
            result = operation(*args, **kwargs)
            
            # Get final metrics
            end_time = time.time()
            end_memory = process.memory_info().rss / 1024 / 1024  # MB
            end_cpu = process.cpu_percent()
            
            # Calculate metrics
            metrics = {
                'execution_time_seconds': end_time - start_time,
                'memory_usage_mb': end_memory - start_memory,
                'cpu_usage_percent': end_cpu - start_cpu,
                'peak_memory_mb': end_memory
            }
            
            return result, metrics
            
        except Exception as e:
            logger.error(f"Performance monitoring failed: {e}")
            result = operation(*args, **kwargs)
            return result, {}
    
    def get_optimization_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Get recommendations for optimizing DataFrame processing."""
        recommendations = []
        
        # Size-based recommendations
        memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        
        if memory_usage > 100:
            recommendations.append("Consider using sampling for large datasets")
        
        if len(df) > 100000:
            recommendations.append("Enable parallel processing for better performance")
        
        # Data type recommendations
        object_columns = df.select_dtypes(include=['object']).columns
        if len(object_columns) > 0:
            recommendations.append("Optimize data types for string columns")
        
        # Memory optimization
        if memory_usage > 50:
            recommendations.append("Apply memory optimization techniques")
        
        # Caching recommendations
        if len(df) > 10000:
            recommendations.append("Enable caching for repeated operations")
        
        return recommendations
    
    def estimate_processing_time(self, df: pd.DataFrame, 
                               operation_type: str = "profiling") -> float:
        """Estimate processing time for an operation."""
        try:
            # Base time estimates (seconds per million rows)
            base_times = {
                'profiling': 60,  # 1 minute per million rows
                'sampling': 5,    # 5 seconds per million rows
                'statistical': 30, # 30 seconds per million rows
                'pattern_discovery': 120  # 2 minutes per million rows
            }
            
            base_time = base_times.get(operation_type, 60)
            
            # Calculate based on data size
            million_rows = len(df) / 1000000
            estimated_time = base_time * million_rows
            
            # Adjust for data complexity
            complexity_factor = 1.0
            
            # More columns = more complex
            if len(df.columns) > 50:
                complexity_factor += 0.5
            
            # More object columns = more complex
            object_columns = len(df.select_dtypes(include=['object']).columns)
            if object_columns > 10:
                complexity_factor += 0.3
            
            return estimated_time * complexity_factor
            
        except Exception as e:
            logger.error(f"Time estimation failed: {e}")
            return 60.0  # Default to 1 minute
    
    def cleanup_resources(self) -> None:
        """Clean up resources and force garbage collection."""
        try:
            # Clear cache
            self.get_dataset_statistics.cache_clear()
            
            # Force garbage collection
            gc.collect()
            
            logger.info("Resources cleaned up")
            
        except Exception as e:
            logger.error(f"Resource cleanup failed: {e}")