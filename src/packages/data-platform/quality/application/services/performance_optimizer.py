"""Performance Optimization Utilities for Data Quality Validation.

Provides utilities for optimizing validation performance including
profiling, caching, and adaptive execution strategies.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging
import time
import cProfile
import pstats
from functools import wraps
import pickle
import hashlib
from pathlib import Path
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import gc

from ...domain.entities.validation_rule import QualityRule, ValidationResult
from ...domain.entities.quality_profile import DatasetId

logger = logging.getLogger(__name__)


@dataclass
class PerformanceProfile:
    """Performance profile for validation operations."""
    operation_name: str
    execution_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    records_processed: int
    throughput_records_per_second: float
    optimization_opportunities: List[str] = field(default_factory=list)
    bottlenecks: List[str] = field(default_factory=list)
    
    def get_efficiency_score(self) -> float:
        """Calculate efficiency score (0-1)."""
        # Simple efficiency calculation based on throughput and resource usage
        if self.throughput_records_per_second == 0:
            return 0.0
        
        # Normalize throughput (assume 1000 records/sec is baseline)
        throughput_score = min(self.throughput_records_per_second / 1000, 1.0)
        
        # Penalty for high resource usage
        memory_penalty = max(0, (self.memory_usage_mb - 512) / 512)  # Penalty above 512MB
        cpu_penalty = max(0, (self.cpu_usage_percent - 80) / 20)  # Penalty above 80%
        
        efficiency = throughput_score * (1 - memory_penalty * 0.3) * (1 - cpu_penalty * 0.2)
        return max(0.0, min(1.0, efficiency))


class PerformanceProfiler:
    """Profile validation performance."""
    
    def __init__(self, enable_detailed_profiling: bool = False):
        self.enable_detailed_profiling = enable_detailed_profiling
        self.profiles: List[PerformanceProfile] = []
        self.profiler = None
    
    def profile_operation(self, operation_name: str):
        """Decorator to profile operations."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return self._profile_function(func, operation_name, *args, **kwargs)
            return wrapper
        return decorator
    
    def _profile_function(self, func: Callable, operation_name: str, *args, **kwargs):
        """Profile a function execution."""
        # Get process for monitoring
        process = psutil.Process()
        
        # Initial measurements
        start_time = time.time()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        start_cpu = process.cpu_percent()
        
        # Enable detailed profiling if requested
        if self.enable_detailed_profiling:
            self.profiler = cProfile.Profile()
            self.profiler.enable()
        
        try:
            # Execute function
            result = func(*args, **kwargs)
            
            # Final measurements
            end_time = time.time()
            end_memory = process.memory_info().rss / 1024 / 1024  # MB
            end_cpu = process.cpu_percent()
            
            # Calculate metrics
            execution_time = end_time - start_time
            memory_usage = max(end_memory - start_memory, 0)
            cpu_usage = (start_cpu + end_cpu) / 2
            
            # Estimate records processed
            records_processed = self._estimate_records_processed(result, args, kwargs)
            throughput = records_processed / execution_time if execution_time > 0 else 0
            
            # Create profile
            profile = PerformanceProfile(
                operation_name=operation_name,
                execution_time=execution_time,
                memory_usage_mb=memory_usage,
                cpu_usage_percent=cpu_usage,
                records_processed=records_processed,
                throughput_records_per_second=throughput
            )
            
            # Analyze for optimization opportunities
            self._analyze_optimization_opportunities(profile)
            
            # Store profile
            self.profiles.append(profile)
            
            return result
            
        finally:
            if self.profiler:
                self.profiler.disable()
    
    def _estimate_records_processed(self, result: Any, args: tuple, kwargs: dict) -> int:
        """Estimate number of records processed."""
        # Try to find DataFrame in arguments
        for arg in args:
            if isinstance(arg, pd.DataFrame):
                return len(arg)
        
        for value in kwargs.values():
            if isinstance(value, pd.DataFrame):
                return len(value)
        
        # Try to infer from result
        if isinstance(result, list):
            if result and hasattr(result[0], 'total_records'):
                return sum(r.total_records for r in result)
        
        return 0
    
    def _analyze_optimization_opportunities(self, profile: PerformanceProfile):
        """Analyze profile for optimization opportunities."""
        # Memory optimization opportunities
        if profile.memory_usage_mb > 1000:
            profile.optimization_opportunities.append("Consider chunking data to reduce memory usage")
        
        # CPU optimization opportunities
        if profile.cpu_usage_percent > 90:
            profile.optimization_opportunities.append("High CPU usage - consider parallel processing")
        
        # Throughput optimization opportunities
        if profile.throughput_records_per_second < 100:
            profile.optimization_opportunities.append("Low throughput - consider vectorization")
        
        # Time-based optimization opportunities
        if profile.execution_time > 60:
            profile.optimization_opportunities.append("Long execution time - consider distributed processing")
        
        # Bottleneck identification
        if profile.memory_usage_mb > 500 and profile.cpu_usage_percent < 50:
            profile.bottlenecks.append("Memory bottleneck - memory usage high but CPU underutilized")
        
        if profile.cpu_usage_percent > 80 and profile.memory_usage_mb < 200:
            profile.bottlenecks.append("CPU bottleneck - CPU usage high but memory usage low")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of performance profiles."""
        if not self.profiles:
            return {'total_operations': 0}
        
        total_time = sum(p.execution_time for p in self.profiles)
        total_memory = sum(p.memory_usage_mb for p in self.profiles)
        total_records = sum(p.records_processed for p in self.profiles)
        
        avg_throughput = sum(p.throughput_records_per_second for p in self.profiles) / len(self.profiles)
        avg_efficiency = sum(p.get_efficiency_score() for p in self.profiles) / len(self.profiles)
        
        return {
            'total_operations': len(self.profiles),
            'total_execution_time': total_time,
            'total_memory_usage_mb': total_memory,
            'total_records_processed': total_records,
            'average_throughput': avg_throughput,
            'average_efficiency': avg_efficiency,
            'optimization_opportunities': self._get_common_optimizations(),
            'bottlenecks': self._get_common_bottlenecks()
        }
    
    def _get_common_optimizations(self) -> List[str]:
        """Get most common optimization opportunities."""
        all_optimizations = []
        for profile in self.profiles:
            all_optimizations.extend(profile.optimization_opportunities)
        
        # Count occurrences
        optimization_counts = {}
        for opt in all_optimizations:
            optimization_counts[opt] = optimization_counts.get(opt, 0) + 1
        
        # Return top 5 most common
        return sorted(optimization_counts.keys(), key=lambda x: optimization_counts[x], reverse=True)[:5]
    
    def _get_common_bottlenecks(self) -> List[str]:
        """Get most common bottlenecks."""
        all_bottlenecks = []
        for profile in self.profiles:
            all_bottlenecks.extend(profile.bottlenecks)
        
        # Count occurrences
        bottleneck_counts = {}
        for bottleneck in all_bottlenecks:
            bottleneck_counts[bottleneck] = bottleneck_counts.get(bottleneck, 0) + 1
        
        # Return top 3 most common
        return sorted(bottleneck_counts.keys(), key=lambda x: bottleneck_counts[x], reverse=True)[:3]
    
    def export_detailed_profile(self, filepath: str):
        """Export detailed profiling data."""
        if not self.profiler:
            logger.warning("No detailed profiling data available")
            return
        
        stats = pstats.Stats(self.profiler)
        stats.sort_stats('cumulative')
        
        with open(filepath, 'w') as f:
            stats.print_stats(file=f)


class ValidationCache:
    """Cache for validation results and computations."""
    
    def __init__(self, max_size: int = 1000, ttl_minutes: int = 30):
        self.max_size = max_size
        self.ttl_minutes = ttl_minutes
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, datetime] = {}
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            if key not in self.cache:
                return None
            
            # Check TTL
            if self._is_expired(key):
                self._remove(key)
                return None
            
            # Update access time
            self.access_times[key] = datetime.now()
            
            return self.cache[key]['value']
    
    def put(self, key: str, value: Any):
        """Put value in cache."""
        with self.lock:
            # Remove expired entries
            self._cleanup_expired()
            
            # Remove oldest entries if cache is full
            if len(self.cache) >= self.max_size:
                self._remove_oldest()
            
            # Add new entry
            self.cache[key] = {
                'value': value,
                'created_at': datetime.now()
            }
            self.access_times[key] = datetime.now()
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired."""
        if key not in self.cache:
            return True
        
        created_at = self.cache[key]['created_at']
        return datetime.now() - created_at > timedelta(minutes=self.ttl_minutes)
    
    def _remove(self, key: str):
        """Remove entry from cache."""
        if key in self.cache:
            del self.cache[key]
        if key in self.access_times:
            del self.access_times[key]
    
    def _cleanup_expired(self):
        """Remove expired entries."""
        expired_keys = [key for key in self.cache.keys() if self._is_expired(key)]
        for key in expired_keys:
            self._remove(key)
    
    def _remove_oldest(self):
        """Remove oldest accessed entry."""
        if not self.access_times:
            return
        
        oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        self._remove(oldest_key)
    
    def clear(self):
        """Clear entire cache."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'ttl_minutes': self.ttl_minutes,
                'oldest_entry': min(self.access_times.values()) if self.access_times else None,
                'newest_entry': max(self.access_times.values()) if self.access_times else None
            }


class AdaptiveExecutionStrategy:
    """Adaptive execution strategy based on performance metrics."""
    
    def __init__(self):
        self.execution_history: List[Dict[str, Any]] = []
        self.strategy_performance: Dict[str, List[float]] = {}
        
    def choose_strategy(self, 
                       dataset_size: int,
                       rule_count: int,
                       memory_limit_mb: int,
                       available_workers: int) -> str:
        """Choose optimal execution strategy."""
        # Get dataset characteristics
        dataset_category = self._categorize_dataset(dataset_size, rule_count)
        
        # Get best performing strategy for this category
        best_strategy = self._get_best_strategy(dataset_category)
        
        # Validate strategy is feasible
        if not self._is_strategy_feasible(best_strategy, dataset_size, memory_limit_mb, available_workers):
            best_strategy = self._get_fallback_strategy(dataset_size, memory_limit_mb)
        
        return best_strategy
    
    def _categorize_dataset(self, dataset_size: int, rule_count: int) -> str:
        """Categorize dataset based on size and complexity."""
        if dataset_size < 10000:
            return 'small'
        elif dataset_size < 100000:
            return 'medium'
        elif dataset_size < 1000000:
            return 'large'
        else:
            return 'xlarge'
    
    def _get_best_strategy(self, dataset_category: str) -> str:
        """Get best performing strategy for dataset category."""
        # Default strategies by category
        default_strategies = {
            'small': 'standard',
            'medium': 'parallel',
            'large': 'chunked',
            'xlarge': 'streaming'
        }
        
        # Check if we have performance history
        if dataset_category in self.strategy_performance:
            performances = self.strategy_performance[dataset_category]
            if performances:
                # Find strategy with best average performance
                best_strategy = max(performances, key=lambda x: np.mean(performances[x]))
                return best_strategy
        
        return default_strategies.get(dataset_category, 'standard')
    
    def _is_strategy_feasible(self, 
                            strategy: str, 
                            dataset_size: int, 
                            memory_limit_mb: int, 
                            available_workers: int) -> bool:
        """Check if strategy is feasible given constraints."""
        # Estimate memory requirements
        estimated_memory = self._estimate_memory_requirement(strategy, dataset_size)
        
        if estimated_memory > memory_limit_mb:
            return False
        
        # Check worker requirements
        if strategy in ['parallel', 'distributed'] and available_workers < 2:
            return False
        
        return True
    
    def _estimate_memory_requirement(self, strategy: str, dataset_size: int) -> float:
        """Estimate memory requirement for strategy."""
        # Rough estimates based on strategy
        memory_per_record = 0.001  # 1KB per record baseline
        
        multipliers = {
            'standard': 1.0,
            'parallel': 1.5,
            'chunked': 0.3,
            'streaming': 0.1,
            'distributed': 0.5
        }
        
        multiplier = multipliers.get(strategy, 1.0)
        return dataset_size * memory_per_record * multiplier
    
    def _get_fallback_strategy(self, dataset_size: int, memory_limit_mb: int) -> str:
        """Get fallback strategy when preferred strategy is not feasible."""
        # Estimate if we can fit dataset in memory
        estimated_memory = self._estimate_memory_requirement('standard', dataset_size)
        
        if estimated_memory <= memory_limit_mb * 0.8:
            return 'standard'
        else:
            return 'streaming'
    
    def record_execution(self, 
                        strategy: str, 
                        dataset_size: int, 
                        rule_count: int, 
                        execution_time: float, 
                        memory_usage: float):
        """Record execution performance."""
        dataset_category = self._categorize_dataset(dataset_size, rule_count)
        
        # Calculate performance score
        performance_score = self._calculate_performance_score(
            execution_time, memory_usage, dataset_size
        )
        
        # Store in history
        execution_record = {
            'strategy': strategy,
            'dataset_category': dataset_category,
            'dataset_size': dataset_size,
            'rule_count': rule_count,
            'execution_time': execution_time,
            'memory_usage': memory_usage,
            'performance_score': performance_score,
            'timestamp': datetime.now()
        }
        
        self.execution_history.append(execution_record)
        
        # Update strategy performance tracking
        if dataset_category not in self.strategy_performance:
            self.strategy_performance[dataset_category] = {}
        
        if strategy not in self.strategy_performance[dataset_category]:
            self.strategy_performance[dataset_category][strategy] = []
        
        self.strategy_performance[dataset_category][strategy].append(performance_score)
        
        # Keep only recent history
        if len(self.execution_history) > 1000:
            self.execution_history = self.execution_history[-1000:]
    
    def _calculate_performance_score(self, 
                                   execution_time: float, 
                                   memory_usage: float, 
                                   dataset_size: int) -> float:
        """Calculate performance score."""
        # Throughput component
        throughput = dataset_size / execution_time if execution_time > 0 else 0
        throughput_score = min(throughput / 1000, 1.0)  # Normalize to 1000 records/sec
        
        # Memory efficiency component
        memory_efficiency = 1.0 - min(memory_usage / 1024, 1.0)  # Penalty for > 1GB
        
        # Combined score
        return (throughput_score * 0.7) + (memory_efficiency * 0.3)
    
    def get_strategy_recommendations(self) -> List[Dict[str, Any]]:
        """Get strategy recommendations based on historical performance."""
        recommendations = []
        
        for category, strategies in self.strategy_performance.items():
            if not strategies:
                continue
            
            # Find best and worst performing strategies
            best_strategy = max(strategies, key=lambda x: np.mean(strategies[x]))
            worst_strategy = min(strategies, key=lambda x: np.mean(strategies[x]))
            
            best_score = np.mean(strategies[best_strategy])
            worst_score = np.mean(strategies[worst_strategy])
            
            recommendations.append({
                'dataset_category': category,
                'best_strategy': best_strategy,
                'best_score': best_score,
                'worst_strategy': worst_strategy,
                'worst_score': worst_score,
                'improvement_potential': best_score - worst_score
            })
        
        return recommendations


class ResourceMonitor:
    """Monitor system resources during validation."""
    
    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.monitoring = False
        self.metrics: List[Dict[str, Any]] = []
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start resource monitoring."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_resources(self):
        """Monitor system resources."""
        while self.monitoring:
            try:
                # Get current metrics
                cpu_percent = psutil.cpu_percent(interval=None)
                memory_info = psutil.virtual_memory()
                disk_usage = psutil.disk_usage('/')
                
                metric = {
                    'timestamp': datetime.now(),
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory_info.percent,
                    'memory_available_mb': memory_info.available / 1024 / 1024,
                    'disk_percent': disk_usage.percent,
                    'disk_free_mb': disk_usage.free / 1024 / 1024
                }
                
                self.metrics.append(metric)
                
                # Keep only recent metrics
                if len(self.metrics) > 3600:  # 1 hour at 1 second intervals
                    self.metrics = self.metrics[-3600:]
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {str(e)}")
                time.sleep(self.monitoring_interval)
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """Get resource usage summary."""
        if not self.metrics:
            return {}
        
        cpu_values = [m['cpu_percent'] for m in self.metrics]
        memory_values = [m['memory_percent'] for m in self.metrics]
        
        return {
            'monitoring_duration_seconds': len(self.metrics) * self.monitoring_interval,
            'cpu_usage': {
                'average': np.mean(cpu_values),
                'max': np.max(cpu_values),
                'min': np.min(cpu_values),
                'std': np.std(cpu_values)
            },
            'memory_usage': {
                'average': np.mean(memory_values),
                'max': np.max(memory_values),
                'min': np.min(memory_values),
                'std': np.std(memory_values)
            },
            'peak_memory_usage': max(memory_values),
            'peak_cpu_usage': max(cpu_values),
            'resource_warnings': self._get_resource_warnings()
        }
    
    def _get_resource_warnings(self) -> List[str]:
        """Get resource usage warnings."""
        warnings = []
        
        if not self.metrics:
            return warnings
        
        cpu_values = [m['cpu_percent'] for m in self.metrics]
        memory_values = [m['memory_percent'] for m in self.metrics]
        
        # CPU warnings
        if np.max(cpu_values) > 90:
            warnings.append("High CPU usage detected (>90%)")
        
        if np.mean(cpu_values) > 80:
            warnings.append("Sustained high CPU usage (>80% average)")
        
        # Memory warnings
        if np.max(memory_values) > 85:
            warnings.append("High memory usage detected (>85%)")
        
        if np.mean(memory_values) > 70:
            warnings.append("Sustained high memory usage (>70% average)")
        
        return warnings