"""Scalability testing suite for Phase 2 components."""

from __future__ import annotations

import time
import gc
import psutil
import numpy as np
import numpy.typing as npt
from typing import Any, Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from pathlib import Path
# Optional dependencies
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing


@dataclass
class ScalabilityResult:
    """Result of a scalability test."""
    component_name: str
    test_type: str
    data_sizes: List[Tuple[int, int]]  # (samples, features)
    execution_times: List[float]
    memory_usages: List[float]  # MB
    throughputs: List[float]  # samples/second
    scalability_score: float  # 0-100 scale
    complexity_analysis: Dict[str, Any]
    bottleneck_analysis: Dict[str, Any]
    recommendations: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ScalabilityConfiguration:
    """Configuration for scalability testing."""
    min_samples: int = 100
    max_samples: int = 100000
    size_multipliers: List[float] = field(default_factory=lambda: [1, 2, 5, 10, 20, 50, 100])
    feature_counts: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100])
    memory_limit_gb: float = 8.0
    time_limit_seconds: float = 300.0  # 5 minutes per test
    enable_plotting: bool = True
    plot_output_dir: str = "scalability_plots"
    min_scalability_score: float = 70.0


class ScalabilityTester:
    """Comprehensive scalability testing suite for Phase 2 components.
    
    Provides detailed scalability analysis including:
    - Data size scaling tests
    - Feature dimension scaling tests
    - Memory usage analysis
    - Computational complexity analysis
    - Bottleneck identification
    - Performance projections
    """
    
    def __init__(self, config: Optional[ScalabilityConfiguration] = None):
        """Initialize scalability tester.
        
        Args:
            config: Scalability testing configuration
        """
        self.config = config or ScalabilityConfiguration()
        self.results: List[ScalabilityResult] = []
        
        # Create output directory for plots
        self.plot_dir = Path(self.config.plot_output_dir)
        self.plot_dir.mkdir(exist_ok=True)
        
        # System information
        self.system_info = {
            'cpu_count': multiprocessing.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'available_memory_gb': psutil.virtual_memory().available / (1024**3)
        }
        
        print(f"📈 Scalability tester initialized:")
        print(f"   System memory: {self.system_info['memory_gb']:.1f}GB")
        print(f"   CPU cores: {self.system_info['cpu_count']}")
        print(f"   Memory limit: {self.config.memory_limit_gb}GB")
    
    def test_data_size_scalability(
        self,
        func: Callable,
        component_name: str,
        base_features: int = 10,
        **func_kwargs
    ) -> ScalabilityResult:
        """Test scalability with increasing data size.
        
        Args:
            func: Function to test
            component_name: Name of the component
            base_features: Number of features to use
            **func_kwargs: Additional function arguments
            
        Returns:
            Scalability test result
        """
        print(f"📊 Testing data size scalability for {component_name}...")
        
        data_sizes = []
        execution_times = []
        memory_usages = []
        throughputs = []
        
        # Generate test sizes
        for multiplier in self.config.size_multipliers:
            samples = int(self.config.min_samples * multiplier)
            if samples > self.config.max_samples:
                break
            
            data_size = (samples, base_features)
            data_sizes.append(data_size)
            
            try:
                # Generate test data
                test_data = self._generate_scalability_data(data_size)
                
                # Check memory requirements
                estimated_memory_gb = (test_data.nbytes / (1024**3)) * 3  # Rough estimate
                if estimated_memory_gb > self.config.memory_limit_gb:
                    print(f"   ⚠️  Skipping size {data_size} - estimated memory {estimated_memory_gb:.1f}GB exceeds limit")
                    break
                
                # Measure performance
                gc.collect()
                process = psutil.Process()
                mem_before = process.memory_info().rss / 1024 / 1024
                
                start_time = time.time()
                result = func(test_data, **func_kwargs)
                end_time = time.time()
                
                mem_after = process.memory_info().rss / 1024 / 1024
                
                execution_time = end_time - start_time
                memory_usage = mem_after - mem_before
                throughput = samples / execution_time if execution_time > 0 else 0
                
                execution_times.append(execution_time)
                memory_usages.append(memory_usage)
                throughputs.append(throughput)
                
                print(f"   Size {data_size}: {execution_time:.3f}s, {memory_usage:.1f}MB, {throughput:.0f} samples/s")
                
                # Check time limit
                if execution_time > self.config.time_limit_seconds:
                    print(f"   ⚠️  Time limit exceeded, stopping at size {data_size}")
                    break
                
            except Exception as e:
                print(f"   ❌ Size {data_size} failed: {str(e)}")
                break
        
        # Analyze scalability
        scalability_score, complexity_analysis, bottleneck_analysis = self._analyze_scalability(
            data_sizes, execution_times, memory_usages, throughputs, "data_size"
        )
        
        result = ScalabilityResult(
            component_name=component_name,
            test_type="data_size_scaling",
            data_sizes=data_sizes,
            execution_times=execution_times,
            memory_usages=memory_usages,
            throughputs=throughputs,
            scalability_score=scalability_score,
            complexity_analysis=complexity_analysis,
            bottleneck_analysis=bottleneck_analysis
        )
        
        # Generate recommendations
        self._generate_scalability_recommendations(result)
        
        # Create plot
        if self.config.enable_plotting:
            self._create_scalability_plot(result)
        
        self.results.append(result)
        print(f"   ✅ Data size scalability score: {scalability_score:.1f}/100")
        
        return result
    
    def test_feature_dimension_scalability(
        self,
        func: Callable,
        component_name: str,
        base_samples: int = 1000,
        **func_kwargs
    ) -> ScalabilityResult:
        """Test scalability with increasing feature dimensions.
        
        Args:
            func: Function to test
            component_name: Name of the component
            base_samples: Number of samples to use
            **func_kwargs: Additional function arguments
            
        Returns:
            Scalability test result
        """
        print(f"📊 Testing feature dimension scalability for {component_name}...")
        
        data_sizes = []
        execution_times = []
        memory_usages = []
        throughputs = []
        
        for features in self.config.feature_counts:
            data_size = (base_samples, features)
            data_sizes.append(data_size)
            
            try:
                # Generate test data
                test_data = self._generate_scalability_data(data_size)
                
                # Check memory requirements
                estimated_memory_gb = (test_data.nbytes / (1024**3)) * 3
                if estimated_memory_gb > self.config.memory_limit_gb:
                    print(f"   ⚠️  Skipping size {data_size} - estimated memory {estimated_memory_gb:.1f}GB exceeds limit")
                    break
                
                # Measure performance
                gc.collect()
                process = psutil.Process()
                mem_before = process.memory_info().rss / 1024 / 1024
                
                start_time = time.time()
                result = func(test_data, **func_kwargs)
                end_time = time.time()
                
                mem_after = process.memory_info().rss / 1024 / 1024
                
                execution_time = end_time - start_time
                memory_usage = mem_after - mem_before
                throughput = base_samples / execution_time if execution_time > 0 else 0
                
                execution_times.append(execution_time)
                memory_usages.append(memory_usage)
                throughputs.append(throughput)
                
                print(f"   Features {features}: {execution_time:.3f}s, {memory_usage:.1f}MB, {throughput:.0f} samples/s")
                
                # Check time limit
                if execution_time > self.config.time_limit_seconds:
                    print(f"   ⚠️  Time limit exceeded, stopping at {features} features")
                    break
                
            except Exception as e:
                print(f"   ❌ Features {features} failed: {str(e)}")
                break
        
        # Analyze scalability
        scalability_score, complexity_analysis, bottleneck_analysis = self._analyze_scalability(
            data_sizes, execution_times, memory_usages, throughputs, "feature_dimension"
        )
        
        result = ScalabilityResult(
            component_name=component_name,
            test_type="feature_dimension_scaling",
            data_sizes=data_sizes,
            execution_times=execution_times,
            memory_usages=memory_usages,
            throughputs=throughputs,
            scalability_score=scalability_score,
            complexity_analysis=complexity_analysis,
            bottleneck_analysis=bottleneck_analysis
        )
        
        # Generate recommendations
        self._generate_scalability_recommendations(result)
        
        # Create plot
        if self.config.enable_plotting:
            self._create_scalability_plot(result)
        
        self.results.append(result)
        print(f"   ✅ Feature dimension scalability score: {scalability_score:.1f}/100")
        
        return result
    
    def test_concurrent_scalability(
        self,
        func: Callable,
        component_name: str,
        data_size: Tuple[int, int] = (1000, 10),
        max_workers: Optional[int] = None,
        **func_kwargs
    ) -> ScalabilityResult:
        """Test scalability with concurrent processing.
        
        Args:
            func: Function to test
            component_name: Name of the component
            data_size: Size of test data (samples, features)
            max_workers: Maximum number of workers
            **func_kwargs: Additional function arguments
            
        Returns:
            Scalability test result
        """
        print(f"🔄 Testing concurrent scalability for {component_name}...")
        
        if max_workers is None:
            max_workers = self.system_info['cpu_count']
        
        worker_counts = [1, 2, 4, min(8, max_workers), max_workers]
        data_sizes = [data_size] * len(worker_counts)
        execution_times = []
        memory_usages = []
        throughputs = []
        
        # Generate test data once
        test_data = self._generate_scalability_data(data_size)
        
        for n_workers in worker_counts:
            try:
                gc.collect()
                process = psutil.Process()
                mem_before = process.memory_info().rss / 1024 / 1024
                
                start_time = time.time()
                
                if n_workers == 1:
                    # Sequential execution
                    result = func(test_data, **func_kwargs)
                else:
                    # Parallel execution
                    chunk_size = max(1, len(test_data) // n_workers)
                    data_chunks = [test_data[i:i + chunk_size] for i in range(0, len(test_data), chunk_size)]
                    
                    with ThreadPoolExecutor(max_workers=n_workers) as executor:
                        futures = [executor.submit(func, chunk, **func_kwargs) for chunk in data_chunks]
                        results = [future.result() for future in futures]
                
                end_time = time.time()
                mem_after = process.memory_info().rss / 1024 / 1024
                
                execution_time = end_time - start_time
                memory_usage = mem_after - mem_before
                throughput = data_size[0] / execution_time if execution_time > 0 else 0
                
                execution_times.append(execution_time)
                memory_usages.append(memory_usage)
                throughputs.append(throughput)
                
                speedup = execution_times[0] / execution_time if execution_times else 1.0
                print(f"   {n_workers} workers: {execution_time:.3f}s ({speedup:.2f}x speedup)")
                
            except Exception as e:
                print(f"   ❌ {n_workers} workers failed: {str(e)}")
                execution_times.append(float('inf'))
                memory_usages.append(0)
                throughputs.append(0)
        
        # Analyze concurrent scalability
        scalability_score, complexity_analysis, bottleneck_analysis = self._analyze_concurrent_scalability(
            worker_counts, execution_times, memory_usages, throughputs
        )
        
        result = ScalabilityResult(
            component_name=component_name,
            test_type="concurrent_scaling",
            data_sizes=data_sizes,
            execution_times=execution_times,
            memory_usages=memory_usages,
            throughputs=throughputs,
            scalability_score=scalability_score,
            complexity_analysis=complexity_analysis,
            bottleneck_analysis=bottleneck_analysis
        )
        
        # Add worker count metadata
        result.complexity_analysis["worker_counts"] = worker_counts
        
        # Generate recommendations
        self._generate_scalability_recommendations(result)
        
        # Create plot
        if self.config.enable_plotting:
            self._create_concurrent_scalability_plot(result)
        
        self.results.append(result)
        print(f"   ✅ Concurrent scalability score: {scalability_score:.1f}/100")
        
        return result
    
    def _generate_scalability_data(self, size: Tuple[int, int]) -> npt.NDArray[np.floating]:
        """Generate test data for scalability testing."""
        n_samples, n_features = size
        
        # Generate mostly normal data with some anomalies
        normal_samples = int(n_samples * 0.9)
        anomaly_samples = n_samples - normal_samples
        
        normal_data = np.random.normal(0, 1, (normal_samples, n_features))
        anomaly_data = np.random.normal(3, 1, (anomaly_samples, n_features))
        
        data = np.vstack([normal_data, anomaly_data])
        np.random.shuffle(data)
        
        return data.astype(np.float32)  # Use float32 for memory efficiency
    
    def _analyze_scalability(
        self,
        data_sizes: List[Tuple[int, int]],
        execution_times: List[float],
        memory_usages: List[float],
        throughputs: List[float],
        scaling_type: str
    ) -> Tuple[float, Dict[str, Any], Dict[str, Any]]:
        """Analyze scalability characteristics."""
        if len(data_sizes) < 2:
            return 0.0, {}, {}
        
        # Extract scaling dimension
        if scaling_type == "data_size":
            scale_values = [size[0] for size in data_sizes]  # Sample count
        else:  # feature_dimension
            scale_values = [size[1] for size in data_sizes]  # Feature count
        
        # Fit complexity curves
        complexity_analysis = self._fit_complexity_curves(scale_values, execution_times, memory_usages)
        
        # Identify bottlenecks
        bottleneck_analysis = self._identify_bottlenecks(
            scale_values, execution_times, memory_usages, throughputs
        )
        
        # Calculate scalability score
        scalability_score = self._calculate_scalability_score(
            complexity_analysis, bottleneck_analysis, throughputs
        )
        
        return scalability_score, complexity_analysis, bottleneck_analysis
    
    def _analyze_concurrent_scalability(
        self,
        worker_counts: List[int],
        execution_times: List[float],
        memory_usages: List[float],
        throughputs: List[float]
    ) -> Tuple[float, Dict[str, Any], Dict[str, Any]]:
        """Analyze concurrent scalability characteristics."""
        if len(worker_counts) < 2:
            return 0.0, {}, {}
        
        # Calculate speedups
        base_time = execution_times[0] if execution_times[0] > 0 else 1.0
        speedups = [base_time / t if t > 0 else 0 for t in execution_times]
        
        # Calculate parallel efficiency
        parallel_efficiencies = [speedup / workers if workers > 0 else 0 
                               for speedup, workers in zip(speedups, worker_counts)]
        
        complexity_analysis = {
            "speedups": speedups,
            "parallel_efficiencies": parallel_efficiencies,
            "max_speedup": max(speedups) if speedups else 0,
            "optimal_workers": worker_counts[speedups.index(max(speedups))] if speedups else 1,
            "scalability_ceiling": max(speedups) / max(worker_counts) if worker_counts else 0
        }
        
        # Identify bottlenecks
        bottleneck_analysis = {
            "efficiency_drop": max(parallel_efficiencies) - min(parallel_efficiencies) if parallel_efficiencies else 0,
            "speedup_plateau": self._detect_speedup_plateau(speedups),
            "memory_overhead": max(memory_usages) - min(memory_usages) if memory_usages else 0
        }
        
        # Calculate concurrent scalability score
        max_efficiency = max(parallel_efficiencies) if parallel_efficiencies else 0
        scalability_score = min(100, max_efficiency * 100)
        
        return scalability_score, complexity_analysis, bottleneck_analysis
    
    def _fit_complexity_curves(
        self,
        scale_values: List[int],
        execution_times: List[float],
        memory_usages: List[float]
    ) -> Dict[str, Any]:
        """Fit computational complexity curves."""
        try:
            try:
                import scipy.optimize as opt
                SCIPY_AVAILABLE = True
            except ImportError:
                SCIPY_AVAILABLE = False
                # Return fallback analysis without curve fitting
                return {
                    "time_complexity": ("Unknown", 0.0, []),
                    "memory_complexity": ("Unknown", 0.0, []),
                    "time_growth_rate": self._calculate_growth_rate(execution_times),
                    "memory_growth_rate": self._calculate_growth_rate(memory_usages)
                }
            
            scale_array = np.array(scale_values)
            time_array = np.array(execution_times)
            memory_array = np.array(memory_usages)
            
            # Define complexity functions
            def linear(x, a, b): return a * x + b
            def quadratic(x, a, b, c): return a * x**2 + b * x + c
            def logarithmic(x, a, b): return a * np.log(x) + b
            def loglinear(x, a, b, c): return a * x * np.log(x) + b * x + c
            
            complexity_fits = {}
            
            # Fit time complexity
            functions = [
                ("O(n)", linear),
                ("O(n²)", quadratic),
                ("O(log n)", logarithmic),
                ("O(n log n)", loglinear)
            ]
            
            best_time_fit = None
            best_time_r2 = -1
            
            for name, func in functions:
                try:
                    popt, _ = opt.curve_fit(func, scale_array, time_array, maxfev=1000)
                    predicted = func(scale_array, *popt)
                    r2 = 1 - np.sum((time_array - predicted)**2) / np.sum((time_array - np.mean(time_array))**2)
                    
                    if r2 > best_time_r2:
                        best_time_r2 = r2
                        best_time_fit = (name, r2, popt.tolist())
                        
                except:
                    continue
            
            # Fit memory complexity
            best_memory_fit = None
            best_memory_r2 = -1
            
            for name, func in functions:
                try:
                    popt, _ = opt.curve_fit(func, scale_array, memory_array, maxfev=1000)
                    predicted = func(scale_array, *popt)
                    r2 = 1 - np.sum((memory_array - predicted)**2) / np.sum((memory_array - np.mean(memory_array))**2)
                    
                    if r2 > best_memory_r2:
                        best_memory_r2 = r2
                        best_memory_fit = (name, r2, popt.tolist())
                        
                except:
                    continue
            
            return {
                "time_complexity": best_time_fit,
                "memory_complexity": best_memory_fit,
                "time_growth_rate": self._calculate_growth_rate(execution_times),
                "memory_growth_rate": self._calculate_growth_rate(memory_usages)
            }
            
        except ImportError:
            # Fallback if scipy is not available
            return {
                "time_complexity": ("Unknown", 0.0, []),
                "memory_complexity": ("Unknown", 0.0, []),
                "time_growth_rate": self._calculate_growth_rate(execution_times),
                "memory_growth_rate": self._calculate_growth_rate(memory_usages)
            }
    
    def _calculate_growth_rate(self, values: List[float]) -> float:
        """Calculate average growth rate."""
        if len(values) < 2:
            return 0.0
        
        growth_rates = []
        for i in range(1, len(values)):
            if values[i-1] > 0:
                growth_rate = (values[i] - values[i-1]) / values[i-1]
                growth_rates.append(growth_rate)
        
        return np.mean(growth_rates) if growth_rates else 0.0
    
    def _identify_bottlenecks(
        self,
        scale_values: List[int],
        execution_times: List[float],
        memory_usages: List[float],
        throughputs: List[float]
    ) -> Dict[str, Any]:
        """Identify performance bottlenecks."""
        bottlenecks = {}
        
        # Time bottlenecks
        if len(execution_times) >= 3:
            time_acceleration = []
            for i in range(2, len(execution_times)):
                if execution_times[i-1] > 0:
                    acceleration = (execution_times[i] - execution_times[i-1]) / execution_times[i-1]
                    time_acceleration.append(acceleration)
            
            if time_acceleration:
                avg_acceleration = np.mean(time_acceleration)
                bottlenecks["time_acceleration"] = avg_acceleration
                if avg_acceleration > 0.5:
                    bottlenecks["time_bottleneck"] = "Exponential time growth detected"
        
        # Memory bottlenecks
        if len(memory_usages) >= 2:
            memory_efficiency = []
            for i in range(1, len(memory_usages)):
                scale_ratio = scale_values[i] / scale_values[i-1]
                memory_ratio = memory_usages[i] / memory_usages[i-1] if memory_usages[i-1] > 0 else 0
                if scale_ratio > 0:
                    efficiency = memory_ratio / scale_ratio
                    memory_efficiency.append(efficiency)
            
            if memory_efficiency:
                avg_memory_efficiency = np.mean(memory_efficiency)
                bottlenecks["memory_efficiency"] = avg_memory_efficiency
                if avg_memory_efficiency > 2.0:
                    bottlenecks["memory_bottleneck"] = "Memory usage growing faster than data size"
        
        # Throughput bottlenecks
        if len(throughputs) >= 2:
            throughput_decline = (throughputs[0] - throughputs[-1]) / throughputs[0] if throughputs[0] > 0 else 0
            bottlenecks["throughput_decline"] = throughput_decline
            if throughput_decline > 0.5:
                bottlenecks["throughput_bottleneck"] = "Significant throughput decline with scale"
        
        return bottlenecks
    
    def _detect_speedup_plateau(self, speedups: List[float]) -> bool:
        """Detect if speedup has plateaued."""
        if len(speedups) < 3:
            return False
        
        # Check if last 3 speedups show minimal improvement
        recent_speedups = speedups[-3:]
        max_recent = max(recent_speedups)
        min_recent = min(recent_speedups)
        
        return (max_recent - min_recent) / max_recent < 0.1  # Less than 10% variation
    
    def _calculate_scalability_score(
        self,
        complexity_analysis: Dict[str, Any],
        bottleneck_analysis: Dict[str, Any],
        throughputs: List[float]
    ) -> float:
        """Calculate overall scalability score (0-100)."""
        score = 100.0
        
        # Penalize based on time complexity
        time_complexity = complexity_analysis.get("time_complexity")
        if time_complexity:
            complexity_name = time_complexity[0]
            if "²" in complexity_name:  # Quadratic
                score -= 30
            elif "log" in complexity_name:  # Logarithmic or log-linear
                score -= 10
            elif "n" in complexity_name:  # Linear
                score -= 5
        
        # Penalize based on memory complexity
        memory_complexity = complexity_analysis.get("memory_complexity")
        if memory_complexity:
            complexity_name = memory_complexity[0]
            if "²" in complexity_name:  # Quadratic
                score -= 20
            elif "log" in complexity_name:  # Logarithmic or log-linear
                score -= 5
        
        # Penalize based on bottlenecks
        time_acceleration = bottleneck_analysis.get("time_acceleration", 0)
        if time_acceleration > 0.5:
            score -= 20
        
        memory_efficiency = bottleneck_analysis.get("memory_efficiency", 1.0)
        if memory_efficiency > 2.0:
            score -= 15
        
        throughput_decline = bottleneck_analysis.get("throughput_decline", 0)
        if throughput_decline > 0.3:
            score -= 15
        
        # Bonus for consistent performance
        if len(throughputs) >= 2:
            throughput_consistency = 1 - (np.std(throughputs) / np.mean(throughputs))
            score += throughput_consistency * 10
        
        return max(0.0, min(100.0, score))
    
    def _generate_scalability_recommendations(self, result: ScalabilityResult) -> None:
        """Generate scalability recommendations."""
        recommendations = []
        
        # Based on scalability score
        if result.scalability_score < 50:
            recommendations.append("Poor scalability detected - consider algorithm optimization or architectural changes")
        elif result.scalability_score < 70:
            recommendations.append("Moderate scalability - consider batch processing or parallel optimization")
        else:
            recommendations.append("Good scalability characteristics")
        
        # Based on complexity analysis
        time_complexity = result.complexity_analysis.get("time_complexity")
        if time_complexity and "²" in time_complexity[0]:
            recommendations.append("Quadratic time complexity detected - consider more efficient algorithms")
        
        memory_complexity = result.complexity_analysis.get("memory_complexity")
        if memory_complexity and "²" in memory_complexity[0]:
            recommendations.append("Quadratic memory complexity detected - consider streaming or chunking")
        
        # Based on bottlenecks
        if "time_bottleneck" in result.bottleneck_analysis:
            recommendations.append("Implement caching or memoization to reduce computational overhead")
        
        if "memory_bottleneck" in result.bottleneck_analysis:
            recommendations.append("Use memory-efficient data structures or implement data streaming")
        
        if "throughput_bottleneck" in result.bottleneck_analysis:
            recommendations.append("Consider parallel processing or batch size optimization")
        
        # Concurrent scalability recommendations
        if result.test_type == "concurrent_scaling":
            optimal_workers = result.complexity_analysis.get("optimal_workers", 1)
            if optimal_workers > 1:
                recommendations.append(f"Optimal concurrent performance with {optimal_workers} workers")
            
            if result.complexity_analysis.get("scalability_ceiling", 0) < 0.5:
                recommendations.append("Limited parallelization benefit - algorithm may be inherently sequential")
        
        result.recommendations = recommendations
    
    def _create_scalability_plot(self, result: ScalabilityResult) -> None:
        """Create scalability visualization plots."""
        if not MATPLOTLIB_AVAILABLE:
            print("   ⚠️  Matplotlib not available - skipping plot generation")
            return
            
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            if result.test_type == "data_size_scaling":
                scale_values = [size[0] for size in result.data_sizes]
                scale_label = "Number of Samples"
            else:
                scale_values = [size[1] for size in result.data_sizes]
                scale_label = "Number of Features"
            
            # Execution time plot
            ax1.plot(scale_values, result.execution_times, 'b-o', linewidth=2, markersize=6)
            ax1.set_xlabel(scale_label)
            ax1.set_ylabel('Execution Time (s)')
            ax1.set_title(f'Execution Time Scaling - {result.component_name}')
            ax1.grid(True, alpha=0.3)
            ax1.set_xscale('log')
            ax1.set_yscale('log')
            
            # Memory usage plot
            ax2.plot(scale_values, result.memory_usages, 'r-s', linewidth=2, markersize=6)
            ax2.set_xlabel(scale_label)
            ax2.set_ylabel('Memory Usage (MB)')
            ax2.set_title(f'Memory Usage Scaling - {result.component_name}')
            ax2.grid(True, alpha=0.3)
            ax2.set_xscale('log')
            
            # Throughput plot
            ax3.plot(scale_values, result.throughputs, 'g-^', linewidth=2, markersize=6)
            ax3.set_xlabel(scale_label)
            ax3.set_ylabel('Throughput (samples/s)')
            ax3.set_title(f'Throughput Scaling - {result.component_name}')
            ax3.grid(True, alpha=0.3)
            ax3.set_xscale('log')
            
            # Scalability summary
            ax4.text(0.1, 0.9, f'Scalability Score: {result.scalability_score:.1f}/100', 
                    transform=ax4.transAxes, fontsize=14, fontweight='bold')
            
            # Add complexity analysis
            y_pos = 0.7
            time_complexity = result.complexity_analysis.get("time_complexity")
            if time_complexity:
                ax4.text(0.1, y_pos, f'Time Complexity: {time_complexity[0]}', 
                        transform=ax4.transAxes, fontsize=12)
                y_pos -= 0.1
            
            memory_complexity = result.complexity_analysis.get("memory_complexity")
            if memory_complexity:
                ax4.text(0.1, y_pos, f'Memory Complexity: {memory_complexity[0]}', 
                        transform=ax4.transAxes, fontsize=12)
                y_pos -= 0.1
            
            # Add recommendations
            y_pos -= 0.1
            for i, rec in enumerate(result.recommendations[:3]):  # Show first 3 recommendations
                ax4.text(0.1, y_pos - i*0.08, f'• {rec[:50]}...', 
                        transform=ax4.transAxes, fontsize=10, wrap=True)
            
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.axis('off')
            
            plt.tight_layout()
            
            # Save plot
            plot_filename = f'{result.component_name}_{result.test_type}_scalability.png'
            plot_path = self.plot_dir / plot_filename
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"   📊 Scalability plot saved: {plot_path}")
            
        except ImportError:
            print("   ⚠️  Matplotlib not available - skipping plot generation")
        except Exception as e:
            print(f"   ⚠️  Plot generation failed: {str(e)}")
    
    def _create_concurrent_scalability_plot(self, result: ScalabilityResult) -> None:
        """Create concurrent scalability visualization."""
        if not MATPLOTLIB_AVAILABLE:
            print("   ⚠️  Matplotlib not available - skipping plot generation")
            return
            
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            worker_counts = result.complexity_analysis.get("worker_counts", [])
            speedups = result.complexity_analysis.get("speedups", [])
            efficiencies = result.complexity_analysis.get("parallel_efficiencies", [])
            
            # Speedup plot
            ax1.plot(worker_counts, speedups, 'b-o', linewidth=2, markersize=6, label='Actual')
            ax1.plot(worker_counts, worker_counts, 'r--', alpha=0.7, label='Ideal')
            ax1.set_xlabel('Number of Workers')
            ax1.set_ylabel('Speedup')
            ax1.set_title(f'Parallel Speedup - {result.component_name}')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Parallel efficiency plot
            ax2.plot(worker_counts, efficiencies, 'g-s', linewidth=2, markersize=6)
            ax2.set_xlabel('Number of Workers')
            ax2.set_ylabel('Parallel Efficiency')
            ax2.set_title(f'Parallel Efficiency - {result.component_name}')
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1.1)
            
            # Execution time plot
            ax3.plot(worker_counts, result.execution_times, 'r-^', linewidth=2, markersize=6)
            ax3.set_xlabel('Number of Workers')
            ax3.set_ylabel('Execution Time (s)')
            ax3.set_title(f'Execution Time vs Workers - {result.component_name}')
            ax3.grid(True, alpha=0.3)
            
            # Summary statistics
            max_speedup = max(speedups) if speedups else 0
            optimal_workers = result.complexity_analysis.get("optimal_workers", 1)
            
            ax4.text(0.1, 0.9, f'Scalability Score: {result.scalability_score:.1f}/100', 
                    transform=ax4.transAxes, fontsize=14, fontweight='bold')
            ax4.text(0.1, 0.8, f'Max Speedup: {max_speedup:.2f}x', 
                    transform=ax4.transAxes, fontsize=12)
            ax4.text(0.1, 0.7, f'Optimal Workers: {optimal_workers}', 
                    transform=ax4.transAxes, fontsize=12)
            
            # Add recommendations
            y_pos = 0.5
            for i, rec in enumerate(result.recommendations[:3]):
                ax4.text(0.1, y_pos - i*0.08, f'• {rec[:50]}...', 
                        transform=ax4.transAxes, fontsize=10)
            
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.axis('off')
            
            plt.tight_layout()
            
            # Save plot
            plot_filename = f'{result.component_name}_concurrent_scalability.png'
            plot_path = self.plot_dir / plot_filename
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"   📊 Concurrent scalability plot saved: {plot_path}")
            
        except ImportError:
            print("   ⚠️  Matplotlib not available - skipping plot generation")
        except Exception as e:
            print(f"   ⚠️  Plot generation failed: {str(e)}")
    
    def generate_scalability_report(self) -> Dict[str, Any]:
        """Generate comprehensive scalability report."""
        if not self.results:
            return {"error": "No scalability results available"}
        
        report = {
            "summary": {
                "total_tests": len(self.results),
                "components_tested": len(set(r.component_name for r in self.results)),
                "average_scalability_score": np.mean([r.scalability_score for r in self.results]),
                "min_scalability_score": min(r.scalability_score for r in self.results),
                "max_scalability_score": max(r.scalability_score for r in self.results)
            },
            "by_component": {},
            "scalability_insights": {
                "bottlenecks": [],
                "recommendations": [],
                "complexity_patterns": {}
            },
            "system_info": self.system_info
        }
        
        # Group by component
        by_component = {}
        for result in self.results:
            if result.component_name not in by_component:
                by_component[result.component_name] = []
            by_component[result.component_name].append(result)
        
        # Analyze each component
        for component, results in by_component.items():
            component_analysis = {
                "test_count": len(results),
                "avg_scalability_score": np.mean([r.scalability_score for r in results]),
                "test_types": [r.test_type for r in results],
                "best_test": max(results, key=lambda r: r.scalability_score).test_type,
                "worst_test": min(results, key=lambda r: r.scalability_score).test_type,
                "recommendations": []
            }
            
            # Collect component recommendations
            for result in results:
                component_analysis["recommendations"].extend(result.recommendations)
            
            report["by_component"][component] = component_analysis
        
        # Global insights
        all_recommendations = []
        complexity_patterns = {}
        
        for result in self.results:
            all_recommendations.extend(result.recommendations)
            
            # Collect complexity patterns
            time_complexity = result.complexity_analysis.get("time_complexity")
            if time_complexity:
                pattern = time_complexity[0]
                if pattern not in complexity_patterns:
                    complexity_patterns[pattern] = 0
                complexity_patterns[pattern] += 1
        
        # Find worst performing components
        worst_performers = sorted(self.results, key=lambda r: r.scalability_score)[:3]
        bottlenecks = [
            {
                "component": r.component_name,
                "test_type": r.test_type,
                "scalability_score": r.scalability_score,
                "bottleneck_analysis": r.bottleneck_analysis
            }
            for r in worst_performers
        ]
        
        report["scalability_insights"]["bottlenecks"] = bottlenecks
        report["scalability_insights"]["recommendations"] = list(set(all_recommendations))
        report["scalability_insights"]["complexity_patterns"] = complexity_patterns
        
        return report
    
    def save_scalability_report(self, filename: Optional[str] = None) -> str:
        """Save scalability report to file."""
        report = self.generate_scalability_report()
        
        if filename is None:
            filename = f"scalability_report_{int(time.time())}.json"
        
        filepath = Path(filename)
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📊 Scalability report saved to: {filepath}")
        return str(filepath)