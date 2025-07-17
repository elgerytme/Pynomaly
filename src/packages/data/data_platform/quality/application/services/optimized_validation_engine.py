"""Optimized Validation Engine for Large Datasets.

Enhanced validation engine with advanced performance optimizations including
streaming processing, memory-efficient operations, and adaptive execution strategies.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union, Callable, Iterator, Tuple
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
import logging
import time
import psutil
import gc
from functools import lru_cache
from contextlib import contextmanager
import multiprocessing as mp
from queue import Queue
import threading
from pathlib import Path
import tempfile
import dask.dataframe as dd
import dask
from dask.distributed import Client, LocalCluster

from ...domain.entities.validation_rule import (
    QualityRule, ValidationResult, ValidationError, ValidationLogic,
    RuleId, ValidationId, ValidationStatus, Severity, LogicType
)
from ...domain.entities.quality_profile import DatasetId, JobMetrics
from .validation_engine import ValidationEngine, ValidationEngineConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OptimizedValidationConfig:
    """Configuration for optimized validation engine."""
    # Basic configuration
    enable_parallel_processing: bool = True
    max_workers: int = None  # Auto-detect based on CPU cores
    timeout_seconds: int = 300
    
    # Memory management
    memory_limit_mb: int = 2048
    chunk_size: int = 10000
    adaptive_chunking: bool = True
    enable_memory_monitoring: bool = True
    memory_threshold: float = 0.8  # 80% of memory limit
    
    # Streaming and disk-based processing
    enable_streaming: bool = True
    enable_disk_caching: bool = True
    temp_directory: Optional[str] = None
    max_disk_usage_mb: int = 10240  # 10GB
    
    # Performance optimizations
    enable_vectorization: bool = True
    enable_lazy_evaluation: bool = True
    enable_query_optimization: bool = True
    enable_column_pruning: bool = True
    
    # Adaptive execution
    enable_adaptive_execution: bool = True
    sample_size_for_estimation: int = 1000
    execution_time_threshold: float = 30.0  # seconds
    
    # Distributed processing
    enable_distributed_processing: bool = False
    dask_scheduler_address: Optional[str] = None
    dask_worker_memory: str = "2GB"
    dask_worker_cores: int = 2
    
    # Caching and optimization
    enable_result_caching: bool = True
    cache_ttl_minutes: int = 30
    enable_rule_compilation: bool = True
    
    def __post_init__(self):
        """Validate and set default values."""
        if self.max_workers is None:
            object.__setattr__(self, 'max_workers', min(mp.cpu_count(), 8))
        
        if self.temp_directory is None:
            object.__setattr__(self, 'temp_directory', tempfile.gettempdir())


class OptimizedValidationEngine:
    """Optimized validation engine for large datasets."""
    
    def __init__(self, config: OptimizedValidationConfig = None):
        """Initialize optimized validation engine."""
        self.config = config or OptimizedValidationConfig()
        self.base_engine = ValidationEngine()
        
        # Performance monitoring
        self.performance_metrics = {
            'total_records_processed': 0,
            'total_execution_time': 0.0,
            'memory_usage_peaks': [],
            'disk_usage_peaks': [],
            'optimization_decisions': [],
            'chunk_processing_times': [],
            'parallel_efficiency': 0.0
        }
        
        # Memory management
        self.memory_monitor = MemoryMonitor(self.config.memory_limit_mb)
        
        # Distributed processing setup
        self.dask_client = None
        if self.config.enable_distributed_processing:
            self._setup_distributed_processing()
        
        # Optimization caches
        self.rule_cache = {}
        self.execution_plan_cache = {}
        self.column_statistics_cache = {}
        
        # Streaming components
        self.stream_processor = StreamProcessor(self.config)
        
        logger.info(f"Optimized validation engine initialized with {self.config.max_workers} workers")
    
    def validate_large_dataset(self, 
                              df: pd.DataFrame,
                              rules: List[QualityRule],
                              dataset_id: DatasetId) -> List[ValidationResult]:
        """Validate large dataset with optimizations."""
        start_time = time.time()
        
        try:
            # Analyze dataset characteristics
            dataset_info = self._analyze_dataset(df)
            
            # Choose optimal execution strategy
            execution_strategy = self._choose_execution_strategy(df, rules, dataset_info)
            
            # Execute validation based on strategy
            if execution_strategy == 'streaming':
                results = self._execute_streaming_validation(df, rules, dataset_id)
            elif execution_strategy == 'distributed':
                results = self._execute_distributed_validation(df, rules, dataset_id)
            elif execution_strategy == 'chunked':
                results = self._execute_chunked_validation(df, rules, dataset_id)
            else:
                results = self._execute_standard_validation(df, rules, dataset_id)
            
            # Update performance metrics
            execution_time = time.time() - start_time
            self.performance_metrics['total_records_processed'] += len(df)
            self.performance_metrics['total_execution_time'] += execution_time
            
            logger.info(f"Large dataset validation completed in {execution_time:.2f}s using {execution_strategy} strategy")
            
            return results
            
        except Exception as e:
            logger.error(f"Large dataset validation failed: {str(e)}")
            raise
    
    def _analyze_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze dataset characteristics for optimization."""
        memory_usage = df.memory_usage(deep=True).sum()
        
        return {
            'record_count': len(df),
            'column_count': len(df.columns),
            'memory_usage_mb': memory_usage / 1024 / 1024,
            'data_types': df.dtypes.value_counts().to_dict(),
            'null_percentages': df.isnull().sum() / len(df),
            'estimated_processing_time': self._estimate_processing_time(df)
        }
    
    def _estimate_processing_time(self, df: pd.DataFrame) -> float:
        """Estimate processing time based on dataset characteristics."""
        # Simple estimation based on record count and column count
        base_time_per_record = 0.001  # 1ms per record baseline
        column_complexity_factor = 1 + (len(df.columns) / 100)  # More columns = more complexity
        
        return len(df) * base_time_per_record * column_complexity_factor
    
    def _choose_execution_strategy(self, 
                                 df: pd.DataFrame, 
                                 rules: List[QualityRule], 
                                 dataset_info: Dict[str, Any]) -> str:
        """Choose optimal execution strategy based on dataset characteristics."""
        record_count = dataset_info['record_count']
        memory_usage_mb = dataset_info['memory_usage_mb']
        estimated_time = dataset_info['estimated_processing_time']
        
        # Decision logic
        if self.config.enable_distributed_processing and record_count > 1000000:
            return 'distributed'
        elif self.config.enable_streaming and memory_usage_mb > self.config.memory_limit_mb:
            return 'streaming'
        elif record_count > 100000 or estimated_time > self.config.execution_time_threshold:
            return 'chunked'
        else:
            return 'standard'
    
    def _execute_streaming_validation(self, 
                                    df: pd.DataFrame,
                                    rules: List[QualityRule],
                                    dataset_id: DatasetId) -> List[ValidationResult]:
        """Execute validation using streaming approach."""
        logger.info(f"Using streaming validation for {len(df)} records")
        
        # Initialize result accumulators
        rule_results = {rule.rule_id: {'passed': 0, 'failed': 0, 'errors': []} for rule in rules}
        
        # Process data in chunks
        chunk_size = self.config.chunk_size
        for chunk_start in range(0, len(df), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(df))
            chunk_df = df.iloc[chunk_start:chunk_end]
            
            # Process chunk
            chunk_results = self._process_chunk(chunk_df, rules, dataset_id)
            
            # Accumulate results
            for result in chunk_results:
                rule_id = result.rule_id
                rule_results[rule_id]['passed'] += result.passed_records
                rule_results[rule_id]['failed'] += result.failed_records
                rule_results[rule_id]['errors'].extend(result.error_details[:10])  # Limit errors
            
            # Monitor memory usage
            if self.config.enable_memory_monitoring:
                self._check_memory_usage()
            
            # Garbage collection
            gc.collect()
        
        # Create final results
        return self._create_aggregated_results(rule_results, rules, dataset_id, len(df))
    
    def _execute_distributed_validation(self, 
                                       df: pd.DataFrame,
                                       rules: List[QualityRule],
                                       dataset_id: DatasetId) -> List[ValidationResult]:
        """Execute validation using distributed processing."""
        if not self.dask_client:
            raise RuntimeError("Distributed processing not available")
        
        logger.info(f"Using distributed validation for {len(df)} records")
        
        # Convert to Dask DataFrame
        ddf = dd.from_pandas(df, npartitions=self.config.max_workers * 2)
        
        # Process partitions
        results = []
        for rule in rules:
            future = self.dask_client.submit(self._process_partition, ddf, rule, dataset_id)
            results.append(future)
        
        # Gather results
        final_results = []
        for future in as_completed(results):
            try:
                result = future.result()
                final_results.append(result)
            except Exception as e:
                logger.error(f"Distributed validation failed: {str(e)}")
        
        return final_results
    
    def _execute_chunked_validation(self, 
                                   df: pd.DataFrame,
                                   rules: List[QualityRule],
                                   dataset_id: DatasetId) -> List[ValidationResult]:
        """Execute validation using chunked processing."""
        logger.info(f"Using chunked validation for {len(df)} records")
        
        # Optimize chunk size based on memory
        optimal_chunk_size = self._calculate_optimal_chunk_size(df)
        
        # Process chunks with parallel execution
        chunk_futures = []
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            for chunk_start in range(0, len(df), optimal_chunk_size):
                chunk_end = min(chunk_start + optimal_chunk_size, len(df))
                chunk_df = df.iloc[chunk_start:chunk_end]
                
                future = executor.submit(self._process_chunk, chunk_df, rules, dataset_id)
                chunk_futures.append(future)
        
        # Collect and aggregate results
        all_chunk_results = []
        for future in as_completed(chunk_futures):
            chunk_results = future.result()
            all_chunk_results.extend(chunk_results)
        
        # Aggregate results by rule
        return self._aggregate_chunk_results(all_chunk_results, rules, dataset_id, len(df))
    
    def _execute_standard_validation(self, 
                                    df: pd.DataFrame,
                                    rules: List[QualityRule],
                                    dataset_id: DatasetId) -> List[ValidationResult]:
        """Execute validation using standard approach."""
        logger.info(f"Using standard validation for {len(df)} records")
        return self.base_engine.validate_dataset(df, rules, dataset_id)
    
    def _process_chunk(self, 
                      chunk_df: pd.DataFrame,
                      rules: List[QualityRule],
                      dataset_id: DatasetId) -> List[ValidationResult]:
        """Process a single chunk of data."""
        chunk_start_time = time.time()
        
        # Apply optimizations to chunk
        optimized_chunk = self._optimize_chunk(chunk_df, rules)
        
        # Process chunk with base engine
        chunk_results = self.base_engine.validate_dataset(optimized_chunk, rules, dataset_id)
        
        # Record performance
        processing_time = time.time() - chunk_start_time
        self.performance_metrics['chunk_processing_times'].append(processing_time)
        
        return chunk_results
    
    def _optimize_chunk(self, chunk_df: pd.DataFrame, rules: List[QualityRule]) -> pd.DataFrame:
        """Apply optimizations to chunk."""
        optimized_df = chunk_df.copy()
        
        # Column pruning - only keep columns needed by rules
        if self.config.enable_column_pruning:
            required_columns = set()
            for rule in rules:
                required_columns.update(rule.target_columns)
            
            available_columns = set(chunk_df.columns)
            columns_to_keep = required_columns.intersection(available_columns)
            
            if columns_to_keep and len(columns_to_keep) < len(chunk_df.columns):
                optimized_df = chunk_df[list(columns_to_keep)]
        
        # Data type optimization
        if self.config.enable_vectorization:
            optimized_df = self._optimize_data_types(optimized_df)
        
        return optimized_df
    
    def _optimize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types for better performance."""
        optimized_df = df.copy()
        
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to convert to categorical if low cardinality
                if df[col].nunique() < len(df) * 0.5:
                    optimized_df[col] = df[col].astype('category')
            elif df[col].dtype == 'int64':
                # Downcast integers if possible
                if df[col].min() >= 0:
                    if df[col].max() <= 255:
                        optimized_df[col] = df[col].astype('uint8')
                    elif df[col].max() <= 65535:
                        optimized_df[col] = df[col].astype('uint16')
                    elif df[col].max() <= 4294967295:
                        optimized_df[col] = df[col].astype('uint32')
        
        return optimized_df
    
    def _calculate_optimal_chunk_size(self, df: pd.DataFrame) -> int:
        """Calculate optimal chunk size based on memory constraints."""
        if not self.config.adaptive_chunking:
            return self.config.chunk_size
        
        # Estimate memory per record
        memory_per_record = df.memory_usage(deep=True).sum() / len(df)
        
        # Calculate chunk size to stay within memory limit
        available_memory = self.config.memory_limit_mb * 1024 * 1024 * self.config.memory_threshold
        optimal_chunk_size = int(available_memory / memory_per_record)
        
        # Ensure chunk size is within reasonable bounds
        min_chunk_size = 1000
        max_chunk_size = 100000
        
        return max(min_chunk_size, min(optimal_chunk_size, max_chunk_size))
    
    def _create_aggregated_results(self, 
                                  rule_results: Dict[str, Dict],
                                  rules: List[QualityRule],
                                  dataset_id: DatasetId,
                                  total_records: int) -> List[ValidationResult]:
        """Create aggregated validation results."""
        final_results = []
        
        for rule in rules:
            rule_id = rule.rule_id
            rule_data = rule_results[rule_id]
            
            passed_records = rule_data['passed']
            failed_records = rule_data['failed']
            failure_rate = failed_records / total_records if total_records > 0 else 0.0
            
            # Determine status
            success_criteria = rule.validation_logic.success_criteria
            pass_rate = passed_records / total_records if total_records > 0 else 0.0
            
            if pass_rate >= success_criteria.min_pass_rate:
                status = ValidationStatus.PASSED
            elif pass_rate >= success_criteria.warning_threshold:
                status = ValidationStatus.WARNING
            else:
                status = ValidationStatus.FAILED
            
            result = ValidationResult(
                validation_id=ValidationId(),
                rule_id=rule_id,
                dataset_id=dataset_id,
                status=status,
                passed_records=passed_records,
                failed_records=failed_records,
                failure_rate=failure_rate,
                error_details=rule_data['errors'][:100],  # Limit error details
                execution_time=timedelta(seconds=1.0),  # Placeholder
                validated_at=datetime.now(),
                total_records=total_records
            )
            
            final_results.append(result)
        
        return final_results
    
    def _aggregate_chunk_results(self, 
                               chunk_results: List[ValidationResult],
                               rules: List[QualityRule],
                               dataset_id: DatasetId,
                               total_records: int) -> List[ValidationResult]:
        """Aggregate results from multiple chunks."""
        # Group results by rule
        rule_groups = {}
        for result in chunk_results:
            rule_id = result.rule_id
            if rule_id not in rule_groups:
                rule_groups[rule_id] = []
            rule_groups[rule_id].append(result)
        
        # Aggregate each rule's results
        final_results = []
        for rule in rules:
            rule_id = rule.rule_id
            rule_chunk_results = rule_groups.get(rule_id, [])
            
            if not rule_chunk_results:
                continue
            
            # Sum up results
            total_passed = sum(r.passed_records for r in rule_chunk_results)
            total_failed = sum(r.failed_records for r in rule_chunk_results)
            all_errors = []
            for r in rule_chunk_results:
                all_errors.extend(r.error_details)
            
            failure_rate = total_failed / total_records if total_records > 0 else 0.0
            
            # Determine status
            success_criteria = rule.validation_logic.success_criteria
            pass_rate = total_passed / total_records if total_records > 0 else 0.0
            
            if pass_rate >= success_criteria.min_pass_rate:
                status = ValidationStatus.PASSED
            elif pass_rate >= success_criteria.warning_threshold:
                status = ValidationStatus.WARNING
            else:
                status = ValidationStatus.FAILED
            
            aggregated_result = ValidationResult(
                validation_id=ValidationId(),
                rule_id=rule_id,
                dataset_id=dataset_id,
                status=status,
                passed_records=total_passed,
                failed_records=total_failed,
                failure_rate=failure_rate,
                error_details=all_errors[:100],  # Limit error details
                execution_time=timedelta(seconds=sum(r.execution_time.total_seconds() for r in rule_chunk_results)),
                validated_at=datetime.now(),
                total_records=total_records
            )
            
            final_results.append(aggregated_result)
        
        return final_results
    
    def _process_partition(self, 
                          ddf: dd.DataFrame,
                          rule: QualityRule,
                          dataset_id: DatasetId) -> ValidationResult:
        """Process a Dask DataFrame partition."""
        # Convert partition to pandas for processing
        pdf = ddf.compute()
        
        # Process with base engine
        results = self.base_engine.validate_dataset(pdf, [rule], dataset_id)
        
        return results[0] if results else None
    
    def _check_memory_usage(self):
        """Check current memory usage."""
        current_usage = self.memory_monitor.get_current_usage()
        
        if current_usage > self.config.memory_threshold:
            logger.warning(f"Memory usage high: {current_usage:.2%}")
            gc.collect()
    
    def _setup_distributed_processing(self):
        """Set up distributed processing with Dask."""
        try:
            if self.config.dask_scheduler_address:
                self.dask_client = Client(self.config.dask_scheduler_address)
            else:
                cluster = LocalCluster(
                    n_workers=self.config.max_workers,
                    threads_per_worker=1,
                    memory_limit=self.config.dask_worker_memory,
                    dashboard_address=None
                )
                self.dask_client = Client(cluster)
            
            logger.info("Distributed processing initialized")
            
        except Exception as e:
            logger.warning(f"Failed to initialize distributed processing: {str(e)}")
            self.dask_client = None
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        metrics = self.performance_metrics.copy()
        
        # Calculate derived metrics
        if metrics['total_records_processed'] > 0:
            metrics['records_per_second'] = metrics['total_records_processed'] / metrics['total_execution_time']
        
        if metrics['chunk_processing_times']:
            metrics['average_chunk_time'] = sum(metrics['chunk_processing_times']) / len(metrics['chunk_processing_times'])
            metrics['chunk_time_std'] = np.std(metrics['chunk_processing_times'])
        
        # Memory and system metrics
        metrics['current_memory_usage'] = self.memory_monitor.get_current_usage()
        metrics['system_cpu_count'] = mp.cpu_count()
        metrics['configured_workers'] = self.config.max_workers
        
        return metrics
    
    def cleanup(self):
        """Clean up resources."""
        if self.dask_client:
            self.dask_client.close()
        
        # Clear caches
        self.rule_cache.clear()
        self.execution_plan_cache.clear()
        self.column_statistics_cache.clear()
        
        # Force garbage collection
        gc.collect()


class MemoryMonitor:
    """Monitor memory usage during validation."""
    
    def __init__(self, memory_limit_mb: int):
        self.memory_limit_mb = memory_limit_mb
        self.process = psutil.Process()
        self.peak_usage = 0.0
    
    def get_current_usage(self) -> float:
        """Get current memory usage as percentage of limit."""
        current_mb = self.process.memory_info().rss / 1024 / 1024
        usage_percent = current_mb / self.memory_limit_mb
        self.peak_usage = max(self.peak_usage, usage_percent)
        return usage_percent
    
    def get_peak_usage(self) -> float:
        """Get peak memory usage."""
        return self.peak_usage
    
    def reset_peak(self):
        """Reset peak usage counter."""
        self.peak_usage = 0.0


class StreamProcessor:
    """Process data in streaming fashion."""
    
    def __init__(self, config: OptimizedValidationConfig):
        self.config = config
        self.temp_dir = Path(config.temp_directory)
        self.temp_dir.mkdir(exist_ok=True)
    
    def process_stream(self, 
                      data_stream: Iterator[pd.DataFrame],
                      rules: List[QualityRule],
                      dataset_id: DatasetId) -> List[ValidationResult]:
        """Process data stream."""
        results = []
        
        for chunk in data_stream:
            # Process chunk
            chunk_results = self._process_chunk(chunk, rules, dataset_id)
            results.extend(chunk_results)
            
            # Save intermediate results if needed
            if self.config.enable_disk_caching:
                self._save_intermediate_results(chunk_results)
        
        return results
    
    def _process_chunk(self, 
                      chunk: pd.DataFrame,
                      rules: List[QualityRule],
                      dataset_id: DatasetId) -> List[ValidationResult]:
        """Process a single chunk."""
        # This would use the base validation engine
        # Implementation depends on specific requirements
        return []
    
    def _save_intermediate_results(self, results: List[ValidationResult]):
        """Save intermediate results to disk."""
        # Implementation for saving results
        pass


class ValidationQueryOptimizer:
    """Optimize validation queries for better performance."""
    
    def __init__(self):
        self.optimization_cache = {}
    
    def optimize_rule_expression(self, rule: QualityRule) -> str:
        """Optimize rule expression for better performance."""
        original_expr = rule.validation_logic.expression
        
        # Check cache first
        if original_expr in self.optimization_cache:
            return self.optimization_cache[original_expr]
        
        # Apply optimizations
        optimized_expr = self._apply_optimizations(original_expr, rule)
        
        # Cache result
        self.optimization_cache[original_expr] = optimized_expr
        
        return optimized_expr
    
    def _apply_optimizations(self, expression: str, rule: QualityRule) -> str:
        """Apply various optimizations to expression."""
        optimized = expression
        
        # Vectorization optimizations
        if rule.validation_logic.logic_type == LogicType.PYTHON:
            optimized = self._optimize_python_expression(optimized)
        
        # SQL optimizations
        elif rule.validation_logic.logic_type == LogicType.SQL:
            optimized = self._optimize_sql_expression(optimized)
        
        return optimized
    
    def _optimize_python_expression(self, expression: str) -> str:
        """Optimize Python expressions."""
        # Replace row-wise operations with vectorized ones
        optimizations = {
            'df.apply(': 'df.apply(lambda x: ',  # Placeholder
            'for i in range(len(df))': 'df.index'  # Placeholder
        }
        
        optimized = expression
        for old, new in optimizations.items():
            optimized = optimized.replace(old, new)
        
        return optimized
    
    def _optimize_sql_expression(self, expression: str) -> str:
        """Optimize SQL expressions."""
        # Basic SQL optimizations
        optimized = expression
        
        # Add indexes hints, optimize joins, etc.
        # This is a placeholder for actual SQL optimizations
        
        return optimized