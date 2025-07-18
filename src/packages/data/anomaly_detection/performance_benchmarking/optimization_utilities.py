"""Performance optimization utilities for Phase 2 components."""

from __future__ import annotations

import time
import gc
import numpy as np
import numpy.typing as npt
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from pathlib import Path
import psutil
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings


@dataclass
class OptimizationResult:
    """Result of a performance optimization."""
    optimization_type: str
    component_name: str
    original_time: float
    optimized_time: float
    speedup_factor: float
    original_memory_mb: float
    optimized_memory_mb: float
    memory_reduction_percent: float
    parameters_used: Dict[str, Any]
    success: bool
    recommendations: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class OptimizationConfiguration:
    """Configuration for optimization utilities."""
    enable_parallel_optimization: bool = True
    enable_memory_optimization: bool = True
    enable_algorithm_optimization: bool = True
    max_workers: int = multiprocessing.cpu_count()
    memory_limit_mb: int = 2048
    optimization_timeout: float = 300.0  # 5 minutes
    min_speedup_threshold: float = 1.1  # Minimum 10% improvement
    batch_size_candidates: List[int] = field(default_factory=lambda: [100, 500, 1000, 5000, 10000])
    contamination_candidates: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.1, 0.15, 0.2])


class OptimizationUtilities:
    """Comprehensive optimization utilities for Phase 2 components.
    
    Provides automatic optimization capabilities including:
    - Batch size optimization
    - Memory usage optimization
    - Algorithm parameter tuning
    - Parallel processing optimization
    - Data type optimization
    """
    
    def __init__(self, config: Optional[OptimizationConfiguration] = None):
        """Initialize optimization utilities.
        
        Args:
            config: Optimization configuration
        """
        self.config = config or OptimizationConfiguration()
        self.optimization_results: List[OptimizationResult] = []
        
        # System information
        self.system_info = {
            'cpu_count': multiprocessing.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'available_memory_gb': psutil.virtual_memory().available / (1024**3)
        }
        
        print(f"🔧 Optimization utilities initialized:")
        print(f"   CPU cores: {self.system_info['cpu_count']}")
        print(f"   Total memory: {self.system_info['memory_gb']:.1f}GB")
        print(f"   Available memory: {self.system_info['available_memory_gb']:.1f}GB")
    
    def optimize_batch_size(
        self,
        func: Callable,
        data: npt.NDArray[np.floating],
        component_name: str,
        **func_kwargs
    ) -> OptimizationResult:
        """Optimize batch size for a given function.
        
        Args:
            func: Function to optimize (should accept batch_size parameter)
            data: Input data
            component_name: Name of the component
            **func_kwargs: Additional function arguments
            
        Returns:
            Optimization result
        """
        print(f"⚡ Optimizing batch size for {component_name}...")
        
        best_batch_size = None
        best_time = float('inf')
        original_time = None
        results = {}
        
        for batch_size in self.config.batch_size_candidates:
            if batch_size > len(data):
                continue
            
            try:
                # Measure performance
                start_time = time.time()
                gc.collect()
                
                # Call function with specific batch size
                if 'batch_size' in func.__code__.co_varnames:
                    func(data, batch_size=batch_size, **func_kwargs)
                else:
                    # Try to split data into batches manually
                    for i in range(0, len(data), batch_size):
                        batch = data[i:i + batch_size]
                        func(batch, **func_kwargs)
                
                end_time = time.time()
                execution_time = end_time - start_time
                
                results[batch_size] = execution_time
                
                if original_time is None:
                    original_time = execution_time
                
                if execution_time < best_time:
                    best_time = execution_time
                    best_batch_size = batch_size
                
                print(f"   Batch size {batch_size}: {execution_time:.3f}s")
                
            except Exception as e:
                print(f"   ❌ Batch size {batch_size} failed: {str(e)}")
                continue
        
        # Create optimization result
        if best_batch_size and original_time:
            speedup = original_time / best_time
            
            result = OptimizationResult(
                optimization_type="batch_size",
                component_name=component_name,
                original_time=original_time,
                optimized_time=best_time,
                speedup_factor=speedup,
                original_memory_mb=0,  # Not measured in this optimization
                optimized_memory_mb=0,
                memory_reduction_percent=0,
                parameters_used={"optimal_batch_size": best_batch_size},
                success=speedup >= self.config.min_speedup_threshold
            )
            
            if result.success:
                result.recommendations.append(
                    f"Use batch_size={best_batch_size} for {speedup:.1f}x speedup"
                )
            else:
                result.recommendations.append(
                    f"Batch size optimization showed minimal improvement ({speedup:.2f}x)"
                )
            
            self.optimization_results.append(result)
            print(f"   ✅ Optimal batch size: {best_batch_size} ({speedup:.2f}x speedup)")
            
            return result
        
        # Return failed result
        return OptimizationResult(
            optimization_type="batch_size",
            component_name=component_name,
            original_time=0,
            optimized_time=0,
            speedup_factor=1.0,
            original_memory_mb=0,
            optimized_memory_mb=0,
            memory_reduction_percent=0,
            parameters_used={},
            success=False,
            recommendations=["Batch size optimization failed"]
        )
    
    def optimize_memory_usage(
        self,
        func: Callable,
        data: npt.NDArray[np.floating],
        component_name: str,
        **func_kwargs
    ) -> OptimizationResult:
        """Optimize memory usage for a given function.
        
        Args:
            func: Function to optimize
            data: Input data
            component_name: Name of the component
            **func_kwargs: Additional function arguments
            
        Returns:
            Optimization result
        """
        print(f"🧠 Optimizing memory usage for {component_name}...")
        
        original_memory = 0
        optimized_memory = 0
        original_time = 0
        optimized_time = 0
        
        try:
            # Measure original performance
            gc.collect()
            process = psutil.Process()
            mem_before = process.memory_info().rss / 1024 / 1024
            
            start_time = time.time()
            original_result = func(data, **func_kwargs)
            end_time = time.time()
            
            mem_after = process.memory_info().rss / 1024 / 1024
            original_memory = mem_after - mem_before
            original_time = end_time - start_time
            
            # Try memory optimizations
            optimizations = []
            
            # 1. Data type optimization
            if data.dtype == np.float64:
                optimized_data = data.astype(np.float32)
                optimizations.append(("float32_conversion", optimized_data))
            
            # 2. Memory-efficient copying
            if hasattr(data, 'copy'):
                optimized_data = np.ascontiguousarray(data)
                optimizations.append(("contiguous_array", optimized_data))
            
            # Try each optimization
            best_optimization = None
            best_memory = original_memory
            best_time = original_time
            
            for opt_name, opt_data in optimizations:
                try:
                    gc.collect()
                    mem_before = process.memory_info().rss / 1024 / 1024
                    
                    start_time = time.time()
                    opt_result = func(opt_data, **func_kwargs)
                    end_time = time.time()
                    
                    mem_after = process.memory_info().rss / 1024 / 1024
                    opt_memory = mem_after - mem_before
                    opt_time = end_time - start_time
                    
                    if opt_memory < best_memory:
                        best_memory = opt_memory
                        best_time = opt_time
                        best_optimization = opt_name
                    
                    print(f"   {opt_name}: {opt_memory:.1f}MB ({opt_time:.3f}s)")
                    
                except Exception as e:
                    print(f"   ❌ {opt_name} failed: {str(e)}")
                    continue
            
            # Calculate results
            optimized_memory = best_memory
            optimized_time = best_time
            memory_reduction = ((original_memory - optimized_memory) / original_memory * 100) if original_memory > 0 else 0
            speedup = original_time / optimized_time if optimized_time > 0 else 1.0
            
            result = OptimizationResult(
                optimization_type="memory_usage",
                component_name=component_name,
                original_time=original_time,
                optimized_time=optimized_time,
                speedup_factor=speedup,
                original_memory_mb=original_memory,
                optimized_memory_mb=optimized_memory,
                memory_reduction_percent=memory_reduction,
                parameters_used={"best_optimization": best_optimization} if best_optimization else {},
                success=memory_reduction > 5  # At least 5% memory reduction
            )
            
            if result.success:
                result.recommendations.append(
                    f"Apply {best_optimization} for {memory_reduction:.1f}% memory reduction"
                )
            else:
                result.recommendations.append(
                    "Memory optimization showed minimal improvement"
                )
            
            self.optimization_results.append(result)
            print(f"   ✅ Memory optimization: {memory_reduction:.1f}% reduction")
            
            return result
            
        except Exception as e:
            print(f"   ❌ Memory optimization failed: {str(e)}")
            
            return OptimizationResult(
                optimization_type="memory_usage",
                component_name=component_name,
                original_time=0,
                optimized_time=0,
                speedup_factor=1.0,
                original_memory_mb=0,
                optimized_memory_mb=0,
                memory_reduction_percent=0,
                parameters_used={},
                success=False,
                recommendations=["Memory optimization failed"]
            )
    
    def optimize_parallel_processing(
        self,
        func: Callable,
        data: npt.NDArray[np.floating],
        component_name: str,
        **func_kwargs
    ) -> OptimizationResult:
        """Optimize parallel processing for a given function.
        
        Args:
            func: Function to optimize
            data: Input data
            component_name: Name of the component
            **func_kwargs: Additional function arguments
            
        Returns:
            Optimization result
        """
        print(f"🔄 Optimizing parallel processing for {component_name}...")
        
        try:
            # Measure sequential performance
            gc.collect()
            start_time = time.time()
            sequential_result = func(data, **func_kwargs)
            sequential_time = time.time() - start_time
            
            print(f"   Sequential: {sequential_time:.3f}s")
            
            # Try different worker counts
            worker_counts = [2, 4, min(8, self.config.max_workers), self.config.max_workers]
            best_time = sequential_time
            best_workers = 1
            
            for n_workers in worker_counts:
                if n_workers > self.system_info['cpu_count']:
                    continue
                
                try:
                    # Split data for parallel processing
                    chunk_size = max(1, len(data) // n_workers)
                    data_chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
                    
                    gc.collect()
                    start_time = time.time()
                    
                    # Use ThreadPoolExecutor for I/O bound tasks, ProcessPoolExecutor for CPU bound
                    with ThreadPoolExecutor(max_workers=n_workers) as executor:
                        futures = [executor.submit(func, chunk, **func_kwargs) for chunk in data_chunks]
                        parallel_results = [future.result() for future in futures]
                    
                    parallel_time = time.time() - start_time
                    
                    if parallel_time < best_time:
                        best_time = parallel_time
                        best_workers = n_workers
                    
                    speedup = sequential_time / parallel_time
                    print(f"   {n_workers} workers: {parallel_time:.3f}s ({speedup:.2f}x)")
                    
                except Exception as e:
                    print(f"   ❌ {n_workers} workers failed: {str(e)}")
                    continue
            
            # Calculate final results
            speedup = sequential_time / best_time
            
            result = OptimizationResult(
                optimization_type="parallel_processing",
                component_name=component_name,
                original_time=sequential_time,
                optimized_time=best_time,
                speedup_factor=speedup,
                original_memory_mb=0,  # Not measured
                optimized_memory_mb=0,
                memory_reduction_percent=0,
                parameters_used={"optimal_workers": best_workers},
                success=speedup >= self.config.min_speedup_threshold
            )
            
            if result.success:
                result.recommendations.append(
                    f"Use {best_workers} workers for {speedup:.1f}x speedup"
                )
            else:
                result.recommendations.append(
                    f"Parallel processing showed minimal improvement ({speedup:.2f}x)"
                )
            
            self.optimization_results.append(result)
            print(f"   ✅ Optimal workers: {best_workers} ({speedup:.2f}x speedup)")
            
            return result
            
        except Exception as e:
            print(f"   ❌ Parallel processing optimization failed: {str(e)}")
            
            return OptimizationResult(
                optimization_type="parallel_processing",
                component_name=component_name,
                original_time=0,
                optimized_time=0,
                speedup_factor=1.0,
                original_memory_mb=0,
                optimized_memory_mb=0,
                memory_reduction_percent=0,
                parameters_used={},
                success=False,
                recommendations=["Parallel processing optimization failed"]
            )
    
    def optimize_algorithm_parameters(
        self,
        func: Callable,
        data: npt.NDArray[np.floating],
        component_name: str,
        parameter_ranges: Dict[str, List[Any]],
        **fixed_kwargs
    ) -> OptimizationResult:
        """Optimize algorithm parameters for a given function.
        
        Args:
            func: Function to optimize
            data: Input data
            component_name: Name of the component
            parameter_ranges: Dictionary of parameter names to value lists
            **fixed_kwargs: Fixed function arguments
            
        Returns:
            Optimization result
        """
        print(f"🎯 Optimizing algorithm parameters for {component_name}...")
        
        try:
            # Generate parameter combinations
            import itertools
            
            param_names = list(parameter_ranges.keys())
            param_values = list(parameter_ranges.values())
            
            # Limit combinations to avoid excessive testing
            max_combinations = 50
            all_combinations = list(itertools.product(*param_values))
            
            if len(all_combinations) > max_combinations:
                # Sample random combinations
                import random
                combinations = random.sample(all_combinations, max_combinations)
            else:
                combinations = all_combinations
            
            print(f"   Testing {len(combinations)} parameter combinations...")
            
            best_time = float('inf')
            best_params = {}
            original_time = None
            
            for i, combination in enumerate(combinations):
                params = dict(zip(param_names, combination))
                kwargs = {**fixed_kwargs, **params}
                
                try:
                    gc.collect()
                    start_time = time.time()
                    result = func(data, **kwargs)
                    end_time = time.time()
                    
                    execution_time = end_time - start_time
                    
                    if original_time is None:
                        original_time = execution_time
                    
                    if execution_time < best_time:
                        best_time = execution_time
                        best_params = params.copy()
                    
                    if i % 10 == 0:  # Progress update
                        print(f"   Tested {i+1}/{len(combinations)} combinations...")
                    
                except Exception as e:
                    # Ignore failed parameter combinations
                    continue
            
            if best_params and original_time:
                speedup = original_time / best_time
                
                result = OptimizationResult(
                    optimization_type="algorithm_parameters",
                    component_name=component_name,
                    original_time=original_time,
                    optimized_time=best_time,
                    speedup_factor=speedup,
                    original_memory_mb=0,
                    optimized_memory_mb=0,
                    memory_reduction_percent=0,
                    parameters_used=best_params,
                    success=speedup >= self.config.min_speedup_threshold
                )
                
                if result.success:
                    param_str = ", ".join([f"{k}={v}" for k, v in best_params.items()])
                    result.recommendations.append(
                        f"Use parameters: {param_str} for {speedup:.1f}x speedup"
                    )
                else:
                    result.recommendations.append(
                        f"Parameter optimization showed minimal improvement ({speedup:.2f}x)"
                    )
                
                self.optimization_results.append(result)
                print(f"   ✅ Optimal parameters: {best_params} ({speedup:.2f}x speedup)")
                
                return result
            
        except Exception as e:
            print(f"   ❌ Parameter optimization failed: {str(e)}")
        
        return OptimizationResult(
            optimization_type="algorithm_parameters",
            component_name=component_name,
            original_time=0,
            optimized_time=0,
            speedup_factor=1.0,
            original_memory_mb=0,
            optimized_memory_mb=0,
            memory_reduction_percent=0,
            parameters_used={},
            success=False,
            recommendations=["Parameter optimization failed"]
        )
    
    def run_comprehensive_optimization(
        self,
        func: Callable,
        data: npt.NDArray[np.floating],
        component_name: str,
        include_batch_size: bool = True,
        include_memory: bool = True,
        include_parallel: bool = True,
        include_parameters: bool = False,
        parameter_ranges: Optional[Dict[str, List[Any]]] = None,
        **func_kwargs
    ) -> List[OptimizationResult]:
        """Run comprehensive optimization analysis.
        
        Args:
            func: Function to optimize
            data: Input data
            component_name: Name of the component
            include_batch_size: Whether to optimize batch size
            include_memory: Whether to optimize memory usage
            include_parallel: Whether to optimize parallel processing
            include_parameters: Whether to optimize algorithm parameters
            parameter_ranges: Parameter ranges for optimization
            **func_kwargs: Additional function arguments
            
        Returns:
            List of optimization results
        """
        print(f"🚀 Running comprehensive optimization for {component_name}...")
        
        results = []
        
        if include_batch_size:
            result = self.optimize_batch_size(func, data, component_name, **func_kwargs)
            results.append(result)
        
        if include_memory:
            result = self.optimize_memory_usage(func, data, component_name, **func_kwargs)
            results.append(result)
        
        if include_parallel:
            result = self.optimize_parallel_processing(func, data, component_name, **func_kwargs)
            results.append(result)
        
        if include_parameters and parameter_ranges:
            result = self.optimize_algorithm_parameters(
                func, data, component_name, parameter_ranges, **func_kwargs
            )
            results.append(result)
        
        print(f"✅ Comprehensive optimization completed: {len(results)} optimizations tested")
        
        return results
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of all optimization results."""
        if not self.optimization_results:
            return {"error": "No optimization results available"}
        
        successful_optimizations = [r for r in self.optimization_results if r.success]
        
        summary = {
            "total_optimizations": len(self.optimization_results),
            "successful_optimizations": len(successful_optimizations),
            "success_rate": len(successful_optimizations) / len(self.optimization_results) * 100,
            "average_speedup": np.mean([r.speedup_factor for r in successful_optimizations]) if successful_optimizations else 1.0,
            "max_speedup": max([r.speedup_factor for r in successful_optimizations]) if successful_optimizations else 1.0,
            "average_memory_reduction": np.mean([r.memory_reduction_percent for r in self.optimization_results]),
            "by_type": {},
            "recommendations": []
        }
        
        # Group by optimization type
        by_type = {}
        for result in self.optimization_results:
            opt_type = result.optimization_type
            if opt_type not in by_type:
                by_type[opt_type] = []
            by_type[opt_type].append(result)
        
        for opt_type, results in by_type.items():
            successful = [r for r in results if r.success]
            by_type[opt_type] = {
                "total": len(results),
                "successful": len(successful),
                "average_speedup": np.mean([r.speedup_factor for r in successful]) if successful else 1.0,
                "components": list(set(r.component_name for r in results))
            }
        
        summary["by_type"] = by_type
        
        # Collect all recommendations
        all_recommendations = []
        for result in successful_optimizations:
            all_recommendations.extend(result.recommendations)
        
        summary["recommendations"] = list(set(all_recommendations))
        
        return summary
    
    def save_optimization_report(self, filename: Optional[str] = None) -> str:
        """Save optimization report to file."""
        import json
        
        summary = self.get_optimization_summary()
        
        # Add detailed results
        detailed_report = {
            "summary": summary,
            "detailed_results": [
                {
                    "optimization_type": r.optimization_type,
                    "component_name": r.component_name,
                    "original_time": r.original_time,
                    "optimized_time": r.optimized_time,
                    "speedup_factor": r.speedup_factor,
                    "original_memory_mb": r.original_memory_mb,
                    "optimized_memory_mb": r.optimized_memory_mb,
                    "memory_reduction_percent": r.memory_reduction_percent,
                    "parameters_used": r.parameters_used,
                    "success": r.success,
                    "recommendations": r.recommendations,
                    "timestamp": r.timestamp
                }
                for r in self.optimization_results
            ],
            "system_info": self.system_info
        }
        
        if filename is None:
            filename = f"optimization_report_{int(time.time())}.json"
        
        filepath = Path(filename)
        
        with open(filepath, 'w') as f:
            json.dump(detailed_report, f, indent=2, default=str)
        
        print(f"📊 Optimization report saved to: {filepath}")
        return str(filepath)