"""Performance optimization and caching service for data validation."""

import time
import hashlib
import pickle
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from enum import Enum
import logging
import psutil
import threading
from collections import defaultdict, OrderedDict
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from pydantic import BaseModel, Field

from ...domain.services.validation_engine import ValidationRule, ValidationResult, ValidationContext

logger = logging.getLogger(__name__)


class SamplingStrategy(str, Enum):
    """Data sampling strategies for large datasets."""
    RANDOM = "random"
    STRATIFIED = "stratified"
    SYSTEMATIC = "systematic"
    CLUSTER = "cluster"
    TIME_BASED = "time_based"
    INTELLIGENT = "intelligent"


class CacheStrategy(str, Enum):
    """Caching strategies for validation results."""
    NONE = "none"
    MEMORY = "memory"
    DISK = "disk"
    DISTRIBUTED = "distributed"
    HYBRID = "hybrid"


class OptimizationLevel(str, Enum):
    """Optimization levels for validation processing."""
    NONE = "none"
    BASIC = "basic"
    AGGRESSIVE = "aggressive"
    CUSTOM = "custom"


@dataclass
class PerformanceMetrics:
    """Performance metrics for validation operations."""
    total_execution_time_ms: float
    cpu_time_ms: float
    memory_peak_mb: float
    memory_avg_mb: float
    records_per_second: float
    rules_per_second: float
    cache_hit_rate: float
    parallelization_factor: float
    sampling_ratio: float
    optimization_savings_ms: float


@dataclass
class OptimizationConfig:
    """Configuration for validation optimization."""
    # Sampling configuration
    enable_sampling: bool = True
    sampling_strategy: SamplingStrategy = SamplingStrategy.INTELLIGENT
    max_sample_size: int = 100000
    min_sample_size: int = 1000
    confidence_level: float = 0.95
    margin_of_error: float = 0.05
    
    # Parallelization configuration
    enable_parallel: bool = True
    max_workers: int = 4
    use_multiprocessing: bool = False
    chunk_size: int = 10000
    
    # Caching configuration
    cache_strategy: CacheStrategy = CacheStrategy.MEMORY
    cache_ttl_hours: int = 24
    max_cache_size_mb: int = 500
    cache_compression: bool = True
    
    # Memory optimization
    enable_memory_optimization: bool = True
    max_memory_usage_mb: int = 2000
    garbage_collection_frequency: int = 100
    
    # Early stopping
    enable_early_stopping: bool = True
    early_stop_threshold: float = 0.95  # Stop if 95% of samples pass
    min_samples_before_stopping: int = 1000
    
    # Rule optimization
    enable_rule_reordering: bool = True
    enable_rule_grouping: bool = True
    fail_fast_mode: bool = False


class LRUCache:
    """LRU Cache implementation for validation results."""
    
    def __init__(self, max_size: int, ttl_hours: int = 24):
        self.max_size = max_size
        self.ttl_seconds = ttl_hours * 3600
        self.cache: OrderedDict = OrderedDict()
        self.timestamps: Dict[str, float] = {}
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self.lock:
            if key not in self.cache:
                return None
            
            # Check TTL
            if time.time() - self.timestamps[key] > self.ttl_seconds:
                del self.cache[key]
                del self.timestamps[key]
                return None
            
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache."""
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.max_size:
                    # Remove least recently used
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
                    del self.timestamps[oldest_key]
            
            self.cache[key] = value
            self.timestamps[key] = time.time()
    
    def clear(self) -> None:
        """Clear cache."""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()
    
    def size(self) -> int:
        """Get cache size."""
        return len(self.cache)


class ValidationCache:
    """Comprehensive caching system for validation results."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.memory_cache = LRUCache(
            max_size=1000,
            ttl_hours=config.cache_ttl_hours
        )
        self.hit_count = 0
        self.miss_count = 0
    
    def generate_cache_key(
        self,
        rule_id: str,
        data_hash: str,
        rule_version: str = "1.0"
    ) -> str:
        """Generate cache key for validation result."""
        key_components = [rule_id, data_hash, rule_version]
        key_string = "|".join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get_data_hash(self, df: pd.DataFrame) -> str:
        """Generate hash for DataFrame."""
        # Use a sample of the data for hashing to improve performance
        sample_size = min(1000, len(df))
        sample_df = df.sample(n=sample_size, random_state=42) if len(df) > sample_size else df
        
        # Create hash from column names, dtypes, and sample values
        components = [
            str(sorted(df.columns.tolist())),
            str(df.dtypes.to_dict()),
            str(sample_df.values.tobytes())
        ]
        
        return hashlib.md5("|".join(components).encode()).hexdigest()
    
    def get(self, rule_id: str, data_hash: str) -> Optional[ValidationResult]:
        """Get cached validation result."""
        cache_key = self.generate_cache_key(rule_id, data_hash)
        
        if self.config.cache_strategy == CacheStrategy.NONE:
            return None
        
        result = self.memory_cache.get(cache_key)
        
        if result is not None:
            self.hit_count += 1
            logger.debug(f"Cache hit for rule {rule_id}")
        else:
            self.miss_count += 1
            logger.debug(f"Cache miss for rule {rule_id}")
        
        return result
    
    def put(self, rule_id: str, data_hash: str, result: ValidationResult) -> None:
        """Cache validation result."""
        if self.config.cache_strategy == CacheStrategy.NONE:
            return
        
        cache_key = self.generate_cache_key(rule_id, data_hash)
        self.memory_cache.put(cache_key, result)
        logger.debug(f"Cached result for rule {rule_id}")
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate."""
        total_requests = self.hit_count + self.miss_count
        return self.hit_count / total_requests if total_requests > 0 else 0.0
    
    def clear(self) -> None:
        """Clear all caches."""
        self.memory_cache.clear()
        self.hit_count = 0
        self.miss_count = 0


class DataSampler:
    """Intelligent data sampling for large datasets."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
    
    def calculate_sample_size(
        self,
        population_size: int,
        confidence_level: float = 0.95,
        margin_of_error: float = 0.05
    ) -> int:
        """Calculate statistically significant sample size."""
        if population_size <= self.config.min_sample_size:
            return population_size
        
        # Z-score for confidence level
        z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
        z_score = z_scores.get(confidence_level, 1.96)
        
        # Conservative estimate (p = 0.5)
        p = 0.5
        
        # Sample size calculation
        numerator = (z_score ** 2) * p * (1 - p)
        denominator = margin_of_error ** 2
        
        sample_size = int(numerator / denominator)
        
        # Adjust for finite population
        if sample_size >= population_size:
            return population_size
        
        adjusted_sample_size = int(
            sample_size / (1 + (sample_size - 1) / population_size)
        )
        
        return max(
            min(adjusted_sample_size, self.config.max_sample_size),
            self.config.min_sample_size
        )
    
    def sample_data(
        self,
        df: pd.DataFrame,
        strategy: Optional[SamplingStrategy] = None,
        target_column: Optional[str] = None
    ) -> Tuple[pd.DataFrame, float]:
        """Sample data using specified strategy."""
        if not self.config.enable_sampling or len(df) <= self.config.min_sample_size:
            return df, 1.0
        
        strategy = strategy or self.config.sampling_strategy
        sample_size = self.calculate_sample_size(len(df))
        
        if strategy == SamplingStrategy.RANDOM:
            sample_df = self._random_sample(df, sample_size)
        elif strategy == SamplingStrategy.STRATIFIED:
            sample_df = self._stratified_sample(df, sample_size, target_column)
        elif strategy == SamplingStrategy.SYSTEMATIC:
            sample_df = self._systematic_sample(df, sample_size)
        elif strategy == SamplingStrategy.CLUSTER:
            sample_df = self._cluster_sample(df, sample_size)
        elif strategy == SamplingStrategy.TIME_BASED:
            sample_df = self._time_based_sample(df, sample_size)
        elif strategy == SamplingStrategy.INTELLIGENT:
            sample_df = self._intelligent_sample(df, sample_size)
        else:
            sample_df = self._random_sample(df, sample_size)
        
        sampling_ratio = len(sample_df) / len(df)
        logger.info(f"Sampled {len(sample_df)} records from {len(df)} using {strategy.value} strategy")
        
        return sample_df, sampling_ratio
    
    def _random_sample(self, df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
        """Random sampling."""
        return df.sample(n=min(sample_size, len(df)), random_state=42)
    
    def _stratified_sample(
        self,
        df: pd.DataFrame,
        sample_size: int,
        target_column: Optional[str]
    ) -> pd.DataFrame:
        """Stratified sampling."""
        if target_column is None or target_column not in df.columns:
            return self._random_sample(df, sample_size)
        
        # Group by target column and sample proportionally
        groups = df.groupby(target_column)
        samples = []
        
        for name, group in groups:
            group_size = len(group)
            group_sample_size = int((group_size / len(df)) * sample_size)
            group_sample_size = max(1, min(group_sample_size, group_size))
            
            group_sample = group.sample(n=group_sample_size, random_state=42)
            samples.append(group_sample)
        
        return pd.concat(samples, ignore_index=True)
    
    def _systematic_sample(self, df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
        """Systematic sampling."""
        if sample_size >= len(df):
            return df
        
        step = len(df) // sample_size
        indices = list(range(0, len(df), step))[:sample_size]
        return df.iloc[indices]
    
    def _cluster_sample(self, df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
        """Cluster sampling (simplified - groups by row index clusters)."""
        if sample_size >= len(df):
            return df
        
        # Simple clustering by row index ranges
        cluster_size = max(1, len(df) // 20)  # 20 clusters
        num_clusters = min(20, sample_size // 10)
        
        cluster_starts = np.random.choice(
            range(0, len(df) - cluster_size, cluster_size),
            size=num_clusters,
            replace=False
        )
        
        samples = []
        for start in cluster_starts:
            end = min(start + cluster_size, len(df))
            cluster_sample_size = min(sample_size // num_clusters, end - start)
            cluster = df.iloc[start:end]
            samples.append(cluster.sample(n=cluster_sample_size, random_state=42))
        
        return pd.concat(samples, ignore_index=True)
    
    def _time_based_sample(self, df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
        """Time-based sampling."""
        # Look for datetime columns
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        
        if len(datetime_cols) == 0:
            return self._random_sample(df, sample_size)
        
        # Use first datetime column for time-based sampling
        time_col = datetime_cols[0]
        df_sorted = df.sort_values(time_col)
        
        # Sample evenly across time periods
        step = len(df_sorted) // sample_size
        indices = list(range(0, len(df_sorted), step))[:sample_size]
        
        return df_sorted.iloc[indices]
    
    def _intelligent_sample(self, df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
        """Intelligent sampling that considers data characteristics."""
        # Use stratified sampling if categorical columns exist
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) > 0:
            # Use column with highest cardinality for stratification
            best_col = None
            max_cardinality = 0
            
            for col in categorical_cols:
                cardinality = df[col].nunique()
                if 2 <= cardinality <= 20 and cardinality > max_cardinality:
                    max_cardinality = cardinality
                    best_col = col
            
            if best_col:
                return self._stratified_sample(df, sample_size, best_col)
        
        # Use time-based sampling if datetime columns exist
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        if len(datetime_cols) > 0:
            return self._time_based_sample(df, sample_size)
        
        # Default to random sampling
        return self._random_sample(df, sample_size)


class ValidationOptimizer:
    """Main optimization service for validation operations."""
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        self.cache = ValidationCache(self.config)
        self.sampler = DataSampler(self.config)
        self.performance_metrics: List[PerformanceMetrics] = []
    
    def optimize_rules(self, rules: List[ValidationRule]) -> List[ValidationRule]:
        """Optimize rule execution order and grouping."""
        if not self.config.enable_rule_reordering:
            return rules
        
        optimized_rules = rules.copy()
        
        # Sort rules by estimated execution time (fast rules first)
        rule_priorities = self._calculate_rule_priorities(optimized_rules)
        optimized_rules.sort(key=lambda r: rule_priorities.get(r.rule_id, 0))
        
        # Group compatible rules
        if self.config.enable_rule_grouping:
            optimized_rules = self._group_compatible_rules(optimized_rules)
        
        logger.info(f"Optimized {len(rules)} rules for execution")
        return optimized_rules
    
    def optimize_validation(
        self,
        df: pd.DataFrame,
        rules: List[ValidationRule],
        context: Optional[ValidationContext] = None
    ) -> Tuple[pd.DataFrame, List[ValidationRule], float]:
        """Optimize dataset and rules for validation."""
        start_time = time.time()
        
        # Sample data if needed
        optimized_df, sampling_ratio = self.sampler.sample_data(df)
        
        # Optimize rules
        optimized_rules = self.optimize_rules(rules)
        
        # Memory optimization
        if self.config.enable_memory_optimization:
            self._optimize_memory_usage(optimized_df)
        
        optimization_time = (time.time() - start_time) * 1000
        
        logger.info(
            f"Optimization completed in {optimization_time:.2f}ms. "
            f"Dataset: {len(df)} -> {len(optimized_df)} rows, "
            f"Rules: {len(rules)} optimized"
        )
        
        return optimized_df, optimized_rules, sampling_ratio
    
    def validate_with_optimization(
        self,
        df: pd.DataFrame,
        rules: List[ValidationRule],
        context: Optional[ValidationContext] = None
    ) -> List[ValidationResult]:
        """Run optimized validation with all optimization features."""
        start_time = time.time()
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Optimize data and rules
        optimized_df, optimized_rules, sampling_ratio = self.optimize_validation(df, rules, context)
        
        # Check cache for existing results
        data_hash = self.cache.get_data_hash(optimized_df)
        cached_results = []
        rules_to_execute = []
        
        for rule in optimized_rules:
            cached_result = self.cache.get(rule.rule_id, data_hash)
            if cached_result:
                cached_results.append(cached_result)
            else:
                rules_to_execute.append(rule)
        
        # Execute remaining rules
        if rules_to_execute:
            if self.config.enable_parallel and len(rules_to_execute) > 1:
                new_results = self._parallel_validation(optimized_df, rules_to_execute, context)
            else:
                new_results = self._sequential_validation(optimized_df, rules_to_execute, context)
            
            # Cache new results
            for result in new_results:
                self.cache.put(result.rule_id, data_hash, result)
        else:
            new_results = []
        
        # Combine results
        all_results = cached_results + new_results
        
        # Adjust metrics for sampling
        if sampling_ratio < 1.0:
            all_results = self._adjust_results_for_sampling(all_results, df, sampling_ratio)
        
        # Record performance metrics
        end_time = time.time()
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        metrics = PerformanceMetrics(
            total_execution_time_ms=(end_time - start_time) * 1000,
            cpu_time_ms=process.cpu_times().user * 1000,
            memory_peak_mb=peak_memory,
            memory_avg_mb=(initial_memory + peak_memory) / 2,
            records_per_second=len(optimized_df) / (end_time - start_time),
            rules_per_second=len(optimized_rules) / (end_time - start_time),
            cache_hit_rate=self.cache.get_hit_rate(),
            parallelization_factor=self.config.max_workers if self.config.enable_parallel else 1,
            sampling_ratio=sampling_ratio,
            optimization_savings_ms=0  # Would need baseline to calculate
        )
        
        self.performance_metrics.append(metrics)
        
        logger.info(
            f"Optimized validation completed in {metrics.total_execution_time_ms:.2f}ms. "
            f"Cache hit rate: {metrics.cache_hit_rate:.2%}, "
            f"Processing rate: {metrics.records_per_second:.0f} records/sec"
        )
        
        return all_results
    
    def _calculate_rule_priorities(self, rules: List[ValidationRule]) -> Dict[str, int]:
        """Calculate rule execution priorities."""
        priorities = {}
        
        for rule in rules:
            priority = 0
            
            # Fast rules get higher priority (lower number)
            if rule.category.value in ['data_type', 'completeness']:
                priority += 1
            elif rule.category.value in ['format', 'range']:
                priority += 2
            elif rule.category.value in ['uniqueness', 'consistency']:
                priority += 3
            else:
                priority += 4
            
            # Rules that can stop early get higher priority
            if rule.fail_fast:
                priority -= 1
            
            priorities[rule.rule_id] = priority
        
        return priorities
    
    def _group_compatible_rules(self, rules: List[ValidationRule]) -> List[ValidationRule]:
        """Group compatible rules for batch execution."""
        # For now, just return rules as-is
        # In a more advanced implementation, we could create composite rules
        return rules
    
    def _optimize_memory_usage(self, df: pd.DataFrame) -> None:
        """Optimize DataFrame memory usage."""
        # Downcast numeric types
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        # Convert object columns to category if appropriate
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() / len(df) < 0.5:  # Less than 50% unique values
                df[col] = df[col].astype('category')
    
    def _parallel_validation(
        self,
        df: pd.DataFrame,
        rules: List[ValidationRule],
        context: Optional[ValidationContext]
    ) -> List[ValidationResult]:
        """Execute validation rules in parallel."""
        results = []
        
        executor_class = ProcessPoolExecutor if self.config.use_multiprocessing else ThreadPoolExecutor
        
        with executor_class(max_workers=self.config.max_workers) as executor:
            # Submit rule validation tasks
            future_to_rule = {
                executor.submit(self._validate_single_rule, df.copy(), rule, context): rule
                for rule in rules
            }
            
            # Collect results
            for future in as_completed(future_to_rule):
                rule = future_to_rule[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Early stopping if configured
                    if (self.config.enable_early_stopping and
                        not result.passed and
                        result.metrics.pass_rate < (1 - self.config.early_stop_threshold)):
                        logger.info(f"Early stopping triggered by rule {rule.rule_id}")
                        
                        # Cancel remaining futures
                        for f in future_to_rule:
                            f.cancel()
                        break
                        
                except Exception as e:
                    logger.error(f"Error validating rule {rule.rule_id}: {e}")
        
        return results
    
    def _sequential_validation(
        self,
        df: pd.DataFrame,
        rules: List[ValidationRule],
        context: Optional[ValidationContext]
    ) -> List[ValidationResult]:
        """Execute validation rules sequentially."""
        results = []
        
        for rule in rules:
            try:
                result = self._validate_single_rule(df, rule, context)
                results.append(result)
                
                # Early stopping if configured
                if (self.config.enable_early_stopping and
                    not result.passed and
                    result.metrics.pass_rate < (1 - self.config.early_stop_threshold)):
                    logger.info(f"Early stopping triggered by rule {rule.rule_id}")
                    break
                    
            except Exception as e:
                logger.error(f"Error validating rule {rule.rule_id}: {e}")
        
        return results
    
    def _validate_single_rule(
        self,
        df: pd.DataFrame,
        rule: ValidationRule,
        context: Optional[ValidationContext]
    ) -> ValidationResult:
        """Validate a single rule against the dataset."""
        start_time = time.time()
        
        # Reset rule state
        rule.reset()
        
        # Dataset-level validation
        dataset_passed = rule.validate_dataset(df)
        
        # Record-level validation with chunking for large datasets
        records_passed = 0
        records_processed = 0
        
        chunk_size = self.config.chunk_size
        for chunk_start in range(0, len(df), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(df))
            chunk = df.iloc[chunk_start:chunk_end]
            
            for idx, (_, row) in enumerate(chunk.iterrows()):
                records_processed += 1
                record = row.to_dict()
                
                if rule.validate_record(record, chunk_start + idx):
                    records_passed += 1
                
                # Early stopping for this rule if fail_fast enabled
                if rule.fail_fast and len(rule.get_errors()) > 0:
                    break
            
            # Memory management
            if (self.config.enable_memory_optimization and
                records_processed % self.config.garbage_collection_frequency == 0):
                import gc
                gc.collect()
            
            # Check memory usage
            if self.config.enable_memory_optimization:
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                if memory_mb > self.config.max_memory_usage_mb:
                    logger.warning(f"Memory usage {memory_mb:.1f}MB exceeds limit {self.config.max_memory_usage_mb}MB")
                    break
        
        overall_passed = dataset_passed and (records_passed == records_processed)
        pass_rate = records_passed / records_processed if records_processed > 0 else 1.0
        
        execution_time = (time.time() - start_time) * 1000
        
        from ...domain.services.validation_engine import ValidationMetrics, ValidationResult
        
        metrics = ValidationMetrics(
            total_records=len(df),
            records_processed=records_processed,
            records_passed=records_passed,
            records_failed=records_processed - records_passed,
            pass_rate=pass_rate,
            execution_time_ms=execution_time
        )
        
        return ValidationResult(
            rule_id=rule.rule_id,
            rule_name=rule.name,
            category=rule.category,
            severity=rule.severity,
            passed=overall_passed,
            metrics=metrics,
            errors=rule.get_errors(),
            context=context or ValidationContext(dataset_name="unknown")
        )
    
    def _adjust_results_for_sampling(
        self,
        results: List[ValidationResult],
        original_df: pd.DataFrame,
        sampling_ratio: float
    ) -> List[ValidationResult]:
        """Adjust validation results to account for sampling."""
        adjusted_results = []
        
        for result in results:
            # Create adjusted metrics
            adjusted_metrics = ValidationMetrics(
                total_records=len(original_df),
                records_processed=int(result.metrics.records_processed / sampling_ratio),
                records_passed=int(result.metrics.records_passed / sampling_ratio),
                records_failed=int(result.metrics.records_failed / sampling_ratio),
                pass_rate=result.metrics.pass_rate,  # Pass rate should remain the same
                execution_time_ms=result.metrics.execution_time_ms
            )
            
            # Create adjusted result
            adjusted_result = ValidationResult(
                validation_id=result.validation_id,
                rule_id=result.rule_id,
                rule_name=result.rule_name,
                category=result.category,
                severity=result.severity,
                passed=result.passed,
                metrics=adjusted_metrics,
                errors=result.errors,  # Keep original errors for debugging
                context=result.context,
                executed_at=result.executed_at,
                statistics={
                    **result.statistics,
                    'sampling_ratio': sampling_ratio,
                    'original_dataset_size': len(original_df),
                    'sampled_dataset_size': result.metrics.total_records
                }
            )
            
            adjusted_results.append(adjusted_result)
        
        return adjusted_results
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary."""
        if not self.performance_metrics:
            return {}
        
        recent_metrics = self.performance_metrics[-10:]  # Last 10 runs
        
        return {
            'total_validations': len(self.performance_metrics),
            'avg_execution_time_ms': sum(m.total_execution_time_ms for m in recent_metrics) / len(recent_metrics),
            'avg_memory_usage_mb': sum(m.memory_avg_mb for m in recent_metrics) / len(recent_metrics),
            'avg_records_per_second': sum(m.records_per_second for m in recent_metrics) / len(recent_metrics),
            'avg_cache_hit_rate': sum(m.cache_hit_rate for m in recent_metrics) / len(recent_metrics),
            'avg_sampling_ratio': sum(m.sampling_ratio for m in recent_metrics) / len(recent_metrics),
            'cache_size': self.cache.memory_cache.size(),
            'optimization_config': {
                'sampling_enabled': self.config.enable_sampling,
                'parallel_enabled': self.config.enable_parallel,
                'caching_enabled': self.config.cache_strategy != CacheStrategy.NONE,
                'memory_optimization_enabled': self.config.enable_memory_optimization
            }
        }
    
    def clear_cache(self) -> None:
        """Clear all caches and reset metrics."""
        self.cache.clear()
        self.performance_metrics.clear()
        logger.info("Validation cache and metrics cleared")