"""Advanced performance profiling for Phase 2 components."""

from __future__ import annotations

import time
import psutil
import gc
import cProfile
import pstats
import io
from typing import Any, Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import numpy.typing as npt
# Optional dependency
try:
    from memory_profiler import profile as memory_profile
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False
    memory_profile = None

import tracemalloc
import threading
import queue
from concurrent.futures import ThreadPoolExecutor


@dataclass
class ProfileResult:
    """Result of a performance profile."""
    component_name: str
    profile_type: str
    execution_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    function_calls: int
    memory_peak_mb: float
    memory_trace: Optional[Dict[str, Any]] = None
    cpu_profile: Optional[str] = None
    hotspots: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ProfilingConfiguration:
    """Configuration for performance profiling."""
    enable_cpu_profiling: bool = True
    enable_memory_profiling: bool = True
    enable_line_profiling: bool = False
    enable_call_graph: bool = True
    memory_precision: int = 3
    max_hotspots: int = 10
    sample_interval: float = 0.01
    output_directory: str = "profiling_results"


class PerformanceProfiler:
    """Advanced performance profiler for Phase 2 components.
    
    Provides detailed profiling capabilities including:
    - CPU profiling with function-level analysis
    - Memory profiling with allocation tracking
    - Hotspot detection and analysis
    - Performance recommendations
    """
    
    def __init__(self, config: Optional[ProfilingConfiguration] = None):
        """Initialize performance profiler.
        
        Args:
            config: Profiling configuration
        """
        self.config = config or ProfilingConfiguration()
        self.output_dir = Path(self.config.output_directory)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize system monitoring
        self.process = psutil.Process()
        
        # Performance tracking
        self.active_profiles: Dict[str, Any] = {}
        self.profile_results: List[ProfileResult] = []
    
    def profile_function(
        self,
        func: Callable,
        component_name: str,
        profile_type: str = "function",
        *args,
        **kwargs
    ) -> Tuple[Any, ProfileResult]:
        """Profile a single function execution.
        
        Args:
            func: Function to profile
            component_name: Name of the component
            profile_type: Type of profiling
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Tuple of (function result, profiling result)
        """
        print(f"🔬 Profiling {component_name}.{profile_type}...")
        
        # Initialize profilers
        profilers = self._initialize_profilers()
        
        try:
            # Start profiling
            self._start_profiling(profilers)
            
            # Execute function
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            # Stop profiling and collect results
            profile_data = self._stop_profiling(profilers)
            
            # Create profile result
            profile_result = self._create_profile_result(
                component_name=component_name,
                profile_type=profile_type,
                execution_time=end_time - start_time,
                profile_data=profile_data
            )
            
            # Analyze and generate recommendations
            self._analyze_performance(profile_result)
            
            self.profile_results.append(profile_result)
            
            print(f"   ✅ Profiling completed: {profile_result.execution_time:.3f}s")
            return result, profile_result
            
        except Exception as e:
            print(f"   ❌ Profiling failed: {str(e)}")
            return None, None
    
    def profile_class_methods(
        self,
        obj: Any,
        method_names: List[str],
        component_name: str,
        test_data: Any = None
    ) -> List[ProfileResult]:
        """Profile multiple methods of a class.
        
        Args:
            obj: Object instance to profile
            method_names: List of method names to profile
            component_name: Name of the component
            test_data: Test data for method calls
            
        Returns:
            List of profiling results
        """
        results = []
        
        for method_name in method_names:
            if hasattr(obj, method_name):
                method = getattr(obj, method_name)
                
                try:
                    if test_data is not None:
                        _, profile_result = self.profile_function(
                            func=lambda: method(test_data),
                            component_name=component_name,
                            profile_type=method_name
                        )
                    else:
                        _, profile_result = self.profile_function(
                            func=method,
                            component_name=component_name,
                            profile_type=method_name
                        )
                    
                    if profile_result:
                        results.append(profile_result)
                        
                except Exception as e:
                    print(f"   ⚠️  Failed to profile {method_name}: {str(e)}")
        
        return results
    
    def profile_streaming_performance(
        self,
        stream_func: Callable,
        data_batches: List[Any],
        component_name: str,
        profile_duration: float = 30.0
    ) -> ProfileResult:
        """Profile streaming performance over time.
        
        Args:
            stream_func: Streaming function to profile
            data_batches: List of data batches to process
            component_name: Name of the component
            profile_duration: Duration to profile (seconds)
            
        Returns:
            Streaming profiling result
        """
        print(f"🌊 Profiling streaming performance for {component_name}...")
        
        # Initialize monitoring
        profilers = self._initialize_profilers()
        
        # Start profiling
        self._start_profiling(profilers)
        
        start_time = time.time()
        processed_batches = 0
        total_samples = 0
        
        try:
            # Process batches for specified duration
            batch_index = 0
            while (time.time() - start_time) < profile_duration:
                if batch_index >= len(data_batches):
                    batch_index = 0  # Loop through batches
                
                batch = data_batches[batch_index]
                stream_func(batch)
                
                processed_batches += 1
                total_samples += len(batch) if hasattr(batch, '__len__') else 1
                batch_index += 1
            
            end_time = time.time()
            
            # Stop profiling
            profile_data = self._stop_profiling(profilers)
            
            # Create streaming-specific result
            profile_result = self._create_profile_result(
                component_name=component_name,
                profile_type="streaming",
                execution_time=end_time - start_time,
                profile_data=profile_data
            )
            
            # Add streaming-specific metrics
            profile_result.metadata = {
                "processed_batches": processed_batches,
                "total_samples": total_samples,
                "batches_per_second": processed_batches / (end_time - start_time),
                "samples_per_second": total_samples / (end_time - start_time),
                "profile_duration": profile_duration
            }
            
            self._analyze_streaming_performance(profile_result)
            self.profile_results.append(profile_result)
            
            print(f"   ✅ Streaming profiling completed: {processed_batches} batches, "
                  f"{total_samples} samples in {end_time - start_time:.1f}s")
            
            return profile_result
            
        except Exception as e:
            print(f"   ❌ Streaming profiling failed: {str(e)}")
            return None
    
    def _initialize_profilers(self) -> Dict[str, Any]:
        """Initialize all profilers."""
        profilers = {}
        
        if self.config.enable_cpu_profiling:
            profilers['cpu'] = cProfile.Profile()
        
        if self.config.enable_memory_profiling:
            tracemalloc.start()
            profilers['memory'] = {
                'initial_memory': self.process.memory_info().rss / 1024 / 1024,
                'peak_memory': 0
            }
        
        return profilers
    
    def _start_profiling(self, profilers: Dict[str, Any]) -> None:
        """Start all profilers."""
        if 'cpu' in profilers:
            profilers['cpu'].enable()
        
        if 'memory' in profilers:
            gc.collect()  # Clean up before starting
    
    def _stop_profiling(self, profilers: Dict[str, Any]) -> Dict[str, Any]:
        """Stop all profilers and collect data."""
        profile_data = {}
        
        if 'cpu' in profilers:
            profilers['cpu'].disable()
            
            # Capture CPU profile
            cpu_stream = io.StringIO()
            ps = pstats.Stats(profilers['cpu'], stream=cpu_stream)
            ps.sort_stats('cumulative')
            ps.print_stats()
            
            profile_data['cpu_profile'] = cpu_stream.getvalue()
            profile_data['cpu_stats'] = ps
        
        if 'memory' in profilers:
            current_memory = self.process.memory_info().rss / 1024 / 1024
            
            if tracemalloc.is_tracing():
                current_trace, peak_trace = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                
                profile_data['memory_trace'] = {
                    'current_mb': current_trace / 1024 / 1024,
                    'peak_mb': peak_trace / 1024 / 1024
                }
            
            profile_data['memory_usage'] = current_memory - profilers['memory']['initial_memory']
            profile_data['memory_peak'] = max(
                current_memory,
                profilers['memory']['peak_memory']
            )
        
        return profile_data
    
    def _create_profile_result(
        self,
        component_name: str,
        profile_type: str,
        execution_time: float,
        profile_data: Dict[str, Any]
    ) -> ProfileResult:
        """Create a profile result from collected data."""
        # Extract metrics
        memory_usage = profile_data.get('memory_usage', 0.0)
        memory_peak = profile_data.get('memory_peak', 0.0)
        cpu_profile = profile_data.get('cpu_profile', "")
        
        # Count function calls
        function_calls = 0
        if 'cpu_stats' in profile_data:
            stats = profile_data['cpu_stats']
            function_calls = sum(call_count for _, call_count, _, _, _ in stats.get_stats().values())
        
        # Get CPU usage (approximate)
        cpu_usage = psutil.cpu_percent(interval=0.1)
        
        return ProfileResult(
            component_name=component_name,
            profile_type=profile_type,
            execution_time=execution_time,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage,
            function_calls=function_calls,
            memory_peak_mb=memory_peak,
            memory_trace=profile_data.get('memory_trace'),
            cpu_profile=cpu_profile
        )
    
    def _analyze_performance(self, profile_result: ProfileResult) -> None:
        """Analyze performance and generate recommendations."""
        recommendations = []
        hotspots = []
        
        # Analyze execution time
        if profile_result.execution_time > 5.0:
            recommendations.append(
                f"High execution time ({profile_result.execution_time:.1f}s) - "
                "consider algorithm optimization or batch processing"
            )
        
        # Analyze memory usage
        if profile_result.memory_usage_mb > 500:
            recommendations.append(
                f"High memory usage ({profile_result.memory_usage_mb:.1f}MB) - "
                "consider memory optimization or data streaming"
            )
        
        # Analyze CPU profile for hotspots
        if profile_result.cpu_profile:
            hotspots = self._extract_hotspots(profile_result.cpu_profile)
        
        # Analyze function call count
        if profile_result.function_calls > 100000:
            recommendations.append(
                f"High function call count ({profile_result.function_calls}) - "
                "consider vectorization or algorithm optimization"
            )
        
        profile_result.hotspots = hotspots
        profile_result.recommendations = recommendations
    
    def _analyze_streaming_performance(self, profile_result: ProfileResult) -> None:
        """Analyze streaming performance specifically."""
        recommendations = []
        metadata = profile_result.metadata
        
        # Analyze throughput
        samples_per_second = metadata.get('samples_per_second', 0)
        if samples_per_second < 1000:
            recommendations.append(
                f"Low streaming throughput ({samples_per_second:.0f} samples/s) - "
                "consider batch size optimization or parallel processing"
            )
        
        # Analyze batch processing rate
        batches_per_second = metadata.get('batches_per_second', 0)
        if batches_per_second < 10:
            recommendations.append(
                f"Low batch processing rate ({batches_per_second:.1f} batches/s) - "
                "consider reducing per-batch processing complexity"
            )
        
        profile_result.recommendations.extend(recommendations)
    
    def _extract_hotspots(self, cpu_profile: str) -> List[Dict[str, Any]]:
        """Extract performance hotspots from CPU profile."""
        hotspots = []
        
        try:
            lines = cpu_profile.split('\n')
            in_stats_section = False
            
            for line in lines:
                if 'ncalls' in line and 'tottime' in line:
                    in_stats_section = True
                    continue
                
                if in_stats_section and line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 6:
                        try:
                            ncalls = parts[0]
                            tottime = float(parts[1])
                            cumtime = float(parts[3])
                            filename_func = ' '.join(parts[5:])
                            
                            if tottime > 0.01:  # Only significant hotspots
                                hotspots.append({
                                    'function': filename_func,
                                    'total_time': tottime,
                                    'cumulative_time': cumtime,
                                    'call_count': ncalls,
                                    'avg_time_per_call': tottime / float(ncalls.split('/')[0]) if '/' not in ncalls else 0
                                })
                        except (ValueError, IndexError):
                            continue
                
                if len(hotspots) >= self.config.max_hotspots:
                    break
        
        except Exception:
            pass  # Ignore parsing errors
        
        # Sort by total time
        hotspots.sort(key=lambda x: x['total_time'], reverse=True)
        return hotspots[:self.config.max_hotspots]
    
    def generate_profiling_report(self) -> Dict[str, Any]:
        """Generate comprehensive profiling report."""
        if not self.profile_results:
            return {"error": "No profiling results available"}
        
        report = {
            "summary": {
                "total_profiles": len(self.profile_results),
                "components_profiled": len(set(r.component_name for r in self.profile_results)),
                "total_execution_time": sum(r.execution_time for r in self.profile_results),
                "average_execution_time": np.mean([r.execution_time for r in self.profile_results]),
                "peak_memory_usage": max(r.memory_peak_mb for r in self.profile_results),
                "total_function_calls": sum(r.function_calls for r in self.profile_results)
            },
            "by_component": {},
            "performance_insights": {
                "bottlenecks": [],
                "recommendations": [],
                "hotspots": []
            }
        }
        
        # Group by component
        by_component = {}
        for result in self.profile_results:
            if result.component_name not in by_component:
                by_component[result.component_name] = []
            by_component[result.component_name].append(result)
        
        # Analyze each component
        for component, results in by_component.items():
            component_analysis = {
                "profile_count": len(results),
                "avg_execution_time": np.mean([r.execution_time for r in results]),
                "avg_memory_usage": np.mean([r.memory_usage_mb for r in results]),
                "total_function_calls": sum(r.function_calls for r in results),
                "recommendations": []
            }
            
            # Collect all recommendations
            for result in results:
                component_analysis["recommendations"].extend(result.recommendations)
            
            report["by_component"][component] = component_analysis
        
        # Global performance insights
        all_recommendations = []
        all_hotspots = []
        
        for result in self.profile_results:
            all_recommendations.extend(result.recommendations)
            all_hotspots.extend(result.hotspots)
        
        # Find bottlenecks
        bottlenecks = []
        for result in self.profile_results:
            if result.execution_time > 2.0:
                bottlenecks.append({
                    "component": result.component_name,
                    "profile_type": result.profile_type,
                    "execution_time": result.execution_time,
                    "memory_usage": result.memory_usage_mb
                })
        
        report["performance_insights"]["bottlenecks"] = sorted(
            bottlenecks, key=lambda x: x["execution_time"], reverse=True
        )[:10]
        
        report["performance_insights"]["recommendations"] = list(set(all_recommendations))[:10]
        
        # Top hotspots across all profiles
        hotspot_summary = {}
        for hotspot in all_hotspots:
            func_name = hotspot['function']
            if func_name not in hotspot_summary:
                hotspot_summary[func_name] = {
                    'total_time': 0,
                    'call_count': 0,
                    'appearances': 0
                }
            
            hotspot_summary[func_name]['total_time'] += hotspot['total_time']
            hotspot_summary[func_name]['appearances'] += 1
        
        top_hotspots = sorted(
            hotspot_summary.items(),
            key=lambda x: x[1]['total_time'],
            reverse=True
        )[:10]
        
        report["performance_insights"]["hotspots"] = [
            {"function": func, **data} for func, data in top_hotspots
        ]
        
        return report
    
    def save_profiling_report(self, filename: Optional[str] = None) -> str:
        """Save profiling report to file."""
        import json
        
        report = self.generate_profiling_report()
        
        if filename is None:
            filename = f"profiling_report_{int(time.time())}.json"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📊 Profiling report saved to: {filepath}")
        return str(filepath)
    
    def get_performance_recommendations(self) -> List[str]:
        """Get prioritized performance recommendations."""
        all_recommendations = []
        
        for result in self.profile_results:
            all_recommendations.extend(result.recommendations)
        
        # Count recommendation frequency
        recommendation_counts = {}
        for rec in all_recommendations:
            key = rec.split(' - ')[0]  # First part before the dash
            recommendation_counts[key] = recommendation_counts.get(key, 0) + 1
        
        # Sort by frequency
        sorted_recommendations = sorted(
            recommendation_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [rec for rec, count in sorted_recommendations]