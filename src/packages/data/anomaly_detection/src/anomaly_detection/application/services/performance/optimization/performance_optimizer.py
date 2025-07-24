"""Advanced performance optimization service for anomaly detection systems."""

from __future__ import annotations

import time
import psutil
import threading
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import numpy.typing as npt

from .....domain.entities.detection_result import DetectionResult
from .....infrastructure.logging import get_logger
from .....infrastructure.monitoring import get_metrics_collector

logger = get_logger(__name__)


class OptimizationStrategy(Enum):
    """Available optimization strategies."""
    MEMORY_OPTIMIZATION = "memory_optimization"
    CPU_OPTIMIZATION = "cpu_optimization"
    THROUGHPUT_OPTIMIZATION = "throughput_optimization"
    LATENCY_OPTIMIZATION = "latency_optimization"
    ACCURACY_OPTIMIZATION = "accuracy_optimization"
    RESOURCE_BALANCING = "resource_balancing"
    BATCH_OPTIMIZATION = "batch_optimization"
    CACHE_OPTIMIZATION = "cache_optimization"


@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot."""
    timestamp: datetime
    cpu_usage_percent: float
    memory_usage_mb: float
    memory_usage_percent: float
    throughput_samples_per_second: float
    latency_ms: float
    accuracy_score: Optional[float] = None
    disk_usage_mb: Optional[float] = None
    network_io_mb: Optional[float] = None
    cache_hit_rate: Optional[float] = None
    batch_size: Optional[int] = None
    queue_size: Optional[int] = None


@dataclass
class OptimizationRecommendation:
    """Performance optimization recommendation."""
    strategy: OptimizationStrategy
    description: str
    expected_improvement: str
    implementation_difficulty: str  # "low", "medium", "high"
    estimated_impact_percent: float
    resource_requirements: Dict[str, Any]
    implementation_steps: List[str]
    confidence_score: float
    priority_score: float


@dataclass
class OptimizationResult:
    """Result of performance optimization analysis."""
    timestamp: datetime
    baseline_metrics: PerformanceMetrics
    target_metrics: Optional[PerformanceMetrics]
    recommendations: List[OptimizationRecommendation]
    performance_bottlenecks: List[str]
    optimization_score: float  # 0-100, higher is better
    estimated_improvement: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceProfiler:
    """System resource profiler for performance analysis."""
    
    def __init__(self):
        self.process = psutil.Process()
        self._start_time: Optional[float] = None
        self._start_cpu_time: Optional[float] = None
        self._samples_processed = 0
        
    def start_profiling(self) -> None:
        """Start performance profiling."""
        self._start_time = time.time()
        self._start_cpu_time = self.process.cpu_times().user + self.process.cpu_times().system
        self._samples_processed = 0
    
    def record_samples(self, count: int) -> None:
        """Record number of samples processed."""
        self._samples_processed += count
    
    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        current_time = time.time()
        
        # CPU and memory usage
        cpu_percent = self.process.cpu_percent()
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        try:
            memory_percent = self.process.memory_percent()
        except:
            memory_percent = 0.0
        
        # Calculate throughput and latency
        if self._start_time is not None:
            elapsed_time = current_time - self._start_time
            throughput = self._samples_processed / elapsed_time if elapsed_time > 0 else 0
            latency = (elapsed_time * 1000) / max(1, self._samples_processed)  # ms per sample
        else:
            throughput = 0
            latency = 0
        
        return PerformanceMetrics(
            timestamp=datetime.utcnow(),
            cpu_usage_percent=cpu_percent,
            memory_usage_mb=memory_mb,
            memory_usage_percent=memory_percent,
            throughput_samples_per_second=throughput,
            latency_ms=latency
        )


class PerformanceOptimizer:
    """Advanced performance optimization service for anomaly detection systems."""
    
    def __init__(
        self,
        profiling_enabled: bool = True,
        optimization_threshold: float = 0.7,
        monitoring_interval_seconds: int = 30
    ):
        """Initialize performance optimizer.
        
        Args:
            profiling_enabled: Enable automatic performance profiling
            optimization_threshold: Threshold below which optimizations are recommended (0-1)
            monitoring_interval_seconds: Interval for background monitoring
        """
        self.profiling_enabled = profiling_enabled
        self.optimization_threshold = optimization_threshold
        self.monitoring_interval = monitoring_interval_seconds
        
        # Components
        self.profiler = PerformanceProfiler()
        self.metrics_collector = get_metrics_collector()
        
        # State tracking
        self._performance_history: List[PerformanceMetrics] = []
        self._optimization_cache: Dict[str, OptimizationResult] = {}
        self._active_optimizations: Dict[str, Any] = {}
        self._monitoring_active = False
        self._monitoring_thread: Optional[threading.Thread] = None
        
        # Performance baselines
        self._cpu_baseline = 50.0  # percent
        self._memory_baseline = 1024.0  # MB
        self._throughput_baseline = 100.0  # samples/sec
        self._latency_baseline = 10.0  # ms
        
        logger.info("PerformanceOptimizer initialized",
                   profiling_enabled=profiling_enabled,
                   optimization_threshold=optimization_threshold)
    
    def start_monitoring(self) -> None:
        """Start background performance monitoring."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        
        logger.info("Started performance monitoring",
                   interval_seconds=self.monitoring_interval)
    
    def stop_monitoring(self) -> None:
        """Stop background performance monitoring."""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        
        logger.info("Stopped performance monitoring")
    
    def analyze_performance(
        self,
        detection_results: Optional[List[DetectionResult]] = None,
        input_data_size: Optional[int] = None,
        strategy_focus: Optional[OptimizationStrategy] = None
    ) -> OptimizationResult:
        """Analyze current performance and generate optimization recommendations.
        
        Args:
            detection_results: Recent detection results for analysis
            input_data_size: Size of input data being processed
            strategy_focus: Focus optimization on specific strategy
            
        Returns:
            Optimization analysis result
        """
        logger.info("Starting performance analysis",
                   detection_results_count=len(detection_results) if detection_results else 0,
                   input_data_size=input_data_size,
                   strategy_focus=strategy_focus.value if strategy_focus else None)
        
        # Get current performance metrics
        current_metrics = self.profiler.get_current_metrics()
        self._performance_history.append(current_metrics)
        
        # Keep history bounded
        if len(self._performance_history) > 1000:
            self._performance_history = self._performance_history[-500:]
        
        # Identify performance bottlenecks
        bottlenecks = self._identify_bottlenecks(current_metrics)
        
        # Generate optimization recommendations
        recommendations = self._generate_recommendations(
            current_metrics, bottlenecks, strategy_focus, detection_results
        )
        
        # Calculate optimization score
        optimization_score = self._calculate_optimization_score(current_metrics, bottlenecks)
        
        # Estimate improvements
        estimated_improvements = self._estimate_improvements(recommendations)
        
        result = OptimizationResult(
            timestamp=datetime.utcnow(),
            baseline_metrics=current_metrics,
            target_metrics=None,  # Would be calculated based on recommendations
            recommendations=recommendations,
            performance_bottlenecks=bottlenecks,
            optimization_score=optimization_score,
            estimated_improvement=estimated_improvements,
            metadata={
                "history_length": len(self._performance_history),
                "strategy_focus": strategy_focus.value if strategy_focus else None,
                "input_data_size": input_data_size
            }
        )
        
        # Cache result
        cache_key = f"{strategy_focus}_{hash(str(bottlenecks))}"
        self._optimization_cache[cache_key] = result
        
        logger.info("Performance analysis completed",
                   optimization_score=optimization_score,
                   recommendations_count=len(recommendations),
                   bottlenecks_count=len(bottlenecks))
        
        return result
    
    def optimize_batch_size(
        self,
        current_batch_size: int,
        target_latency_ms: Optional[float] = None,
        target_throughput: Optional[float] = None
    ) -> Tuple[int, str]:
        """Optimize batch size for better performance.
        
        Args:
            current_batch_size: Current batch size
            target_latency_ms: Target latency in milliseconds
            target_throughput: Target throughput in samples/second
            
        Returns:
            Tuple of (optimal_batch_size, reasoning)
        """
        logger.info("Optimizing batch size",
                   current_batch_size=current_batch_size,
                   target_latency_ms=target_latency_ms,
                   target_throughput=target_throughput)
        
        current_metrics = self.profiler.get_current_metrics()
        
        # Simple heuristic-based optimization
        optimal_size = current_batch_size
        reasoning = "No optimization needed"
        
        # Memory-based optimization
        if current_metrics.memory_usage_percent > 80:
            optimal_size = max(1, current_batch_size // 2)
            reasoning = "Reduced batch size due to high memory usage"
        
        # CPU-based optimization
        elif current_metrics.cpu_usage_percent < 30:
            optimal_size = min(current_batch_size * 2, 1000)
            reasoning = "Increased batch size due to low CPU utilization"
        
        # Latency-based optimization
        elif target_latency_ms and current_metrics.latency_ms > target_latency_ms:
            optimal_size = max(1, int(current_batch_size * (target_latency_ms / current_metrics.latency_ms)))
            reasoning = f"Adjusted batch size to meet latency target of {target_latency_ms}ms"
        
        # Throughput-based optimization
        elif target_throughput and current_metrics.throughput_samples_per_second < target_throughput:
            optimal_size = min(int(current_batch_size * 1.5), 1000)
            reasoning = f"Increased batch size to improve throughput toward {target_throughput} samples/sec"
        
        logger.info("Batch size optimization completed",
                   original_size=current_batch_size,
                   optimal_size=optimal_size,
                   reasoning=reasoning)
        
        return optimal_size, reasoning
    
    def optimize_memory_usage(self) -> Dict[str, Any]:
        """Optimize memory usage and return optimization suggestions."""
        current_metrics = self.profiler.get_current_metrics()
        
        suggestions = {
            "current_memory_mb": current_metrics.memory_usage_mb,
            "memory_percent": current_metrics.memory_usage_percent,
            "optimizations": []
        }
        
        if current_metrics.memory_usage_percent > 80:
            suggestions["optimizations"].extend([
                "Reduce batch size to decrease memory footprint",
                "Implement data streaming instead of loading all data at once",
                "Use memory-efficient data structures",
                "Clear intermediate results more frequently"
            ])
        
        if current_metrics.memory_usage_mb > 2000:  # 2GB
            suggestions["optimizations"].extend([
                "Consider data compression",
                "Implement lazy loading for large datasets",
                "Use memory mapping for large files"
            ])
        
        return suggestions
    
    def get_performance_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive performance report.
        
        Args:
            hours: Number of hours of history to analyze
            
        Returns:
            Performance report dictionary
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_history = [
            m for m in self._performance_history 
            if m.timestamp >= cutoff_time
        ]
        
        if not recent_history:
            return {"error": "No performance data available for the specified period"}
        
        # Calculate statistics
        cpu_values = [m.cpu_usage_percent for m in recent_history]
        memory_values = [m.memory_usage_mb for m in recent_history]
        throughput_values = [m.throughput_samples_per_second for m in recent_history if m.throughput_samples_per_second > 0]
        latency_values = [m.latency_ms for m in recent_history if m.latency_ms > 0]
        
        report = {
            "period_hours": hours,
            "data_points": len(recent_history),
            "cpu_stats": {
                "mean": np.mean(cpu_values) if cpu_values else 0,
                "max": np.max(cpu_values) if cpu_values else 0,
                "min": np.min(cpu_values) if cpu_values else 0,
                "std": np.std(cpu_values) if cpu_values else 0
            },
            "memory_stats": {
                "mean_mb": np.mean(memory_values) if memory_values else 0,
                "max_mb": np.max(memory_values) if memory_values else 0,
                "min_mb": np.min(memory_values) if memory_values else 0,
                "growth_rate": self._calculate_growth_rate(memory_values)
            },
            "throughput_stats": {
                "mean_samples_per_sec": np.mean(throughput_values) if throughput_values else 0,
                "max_samples_per_sec": np.max(throughput_values) if throughput_values else 0,
                "min_samples_per_sec": np.min(throughput_values) if throughput_values else 0
            },
            "latency_stats": {
                "mean_ms": np.mean(latency_values) if latency_values else 0,
                "p95_ms": np.percentile(latency_values, 95) if latency_values else 0,
                "p99_ms": np.percentile(latency_values, 99) if latency_values else 0
            },
            "current_metrics": recent_history[-1] if recent_history else None,
            "trend_analysis": self._analyze_trends(recent_history)
        }
        
        return report
    
    def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self._monitoring_active:
            try:
                if self.profiling_enabled:
                    metrics = self.profiler.get_current_metrics()
                    self._performance_history.append(metrics)
                    
                    # Record metrics
                    self.metrics_collector.record_metric(
                        "performance.cpu_usage_percent", 
                        metrics.cpu_usage_percent,
                        {"component": "performance_optimizer"}
                    )
                    self.metrics_collector.record_metric(
                        "performance.memory_usage_mb", 
                        metrics.memory_usage_mb,
                        {"component": "performance_optimizer"}
                    )
                    
                    # Check for performance issues
                    if metrics.cpu_usage_percent > 90:
                        logger.warning("High CPU usage detected",
                                     cpu_percent=metrics.cpu_usage_percent)
                    
                    if metrics.memory_usage_percent > 90:
                        logger.warning("High memory usage detected",
                                     memory_percent=metrics.memory_usage_percent)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error("Error in performance monitoring loop", error=str(e))
                time.sleep(self.monitoring_interval)
    
    def _identify_bottlenecks(self, metrics: PerformanceMetrics) -> List[str]:
        """Identify performance bottlenecks."""
        bottlenecks = []
        
        if metrics.cpu_usage_percent > self._cpu_baseline:
            bottlenecks.append("High CPU usage")
        
        if metrics.memory_usage_mb > self._memory_baseline:
            bottlenecks.append("High memory usage")
        
        if metrics.throughput_samples_per_second < self._throughput_baseline:
            bottlenecks.append("Low throughput")
        
        if metrics.latency_ms > self._latency_baseline:
            bottlenecks.append("High latency")
        
        if metrics.memory_usage_percent > 80:
            bottlenecks.append("Memory pressure")
        
        return bottlenecks
    
    def _generate_recommendations(
        self,
        metrics: PerformanceMetrics,
        bottlenecks: List[str],
        strategy_focus: Optional[OptimizationStrategy],
        detection_results: Optional[List[DetectionResult]]
    ) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # Memory optimization
        if "High memory usage" in bottlenecks or strategy_focus == OptimizationStrategy.MEMORY_OPTIMIZATION:
            recommendations.append(OptimizationRecommendation(
                strategy=OptimizationStrategy.MEMORY_OPTIMIZATION,
                description="Reduce memory consumption through efficient data structures and streaming",
                expected_improvement="20-40% reduction in memory usage",
                implementation_difficulty="medium",
                estimated_impact_percent=30.0,
                resource_requirements={"development_time": "2-3 days"},
                implementation_steps=[
                    "Implement data streaming for large datasets",
                    "Use memory-efficient data structures",
                    "Add garbage collection optimization",
                    "Implement memory pooling"
                ],
                confidence_score=0.8,
                priority_score=0.9 if "Memory pressure" in bottlenecks else 0.6
            ))
        
        # CPU optimization
        if "High CPU usage" in bottlenecks or strategy_focus == OptimizationStrategy.CPU_OPTIMIZATION:
            recommendations.append(OptimizationRecommendation(
                strategy=OptimizationStrategy.CPU_OPTIMIZATION,
                description="Optimize CPU usage through parallelization and algorithm improvements",
                expected_improvement="15-30% reduction in CPU usage",
                implementation_difficulty="medium",
                estimated_impact_percent=25.0,
                resource_requirements={"development_time": "3-5 days"},
                implementation_steps=[
                    "Implement parallel processing",
                    "Optimize algorithm complexity",
                    "Use vectorized operations",
                    "Add CPU-efficient caching"
                ],
                confidence_score=0.75,
                priority_score=0.8
            ))
        
        # Throughput optimization
        if "Low throughput" in bottlenecks or strategy_focus == OptimizationStrategy.THROUGHPUT_OPTIMIZATION:
            recommendations.append(OptimizationRecommendation(
                strategy=OptimizationStrategy.THROUGHPUT_OPTIMIZATION,
                description="Increase processing throughput through batching and pipelining",
                expected_improvement="50-100% increase in throughput",
                implementation_difficulty="high",
                estimated_impact_percent=60.0,
                resource_requirements={"development_time": "5-7 days"},
                implementation_steps=[
                    "Implement batch processing",
                    "Add processing pipelines",
                    "Optimize I/O operations",
                    "Use asynchronous processing"
                ],
                confidence_score=0.85,
                priority_score=0.9
            ))
        
        # Latency optimization
        if "High latency" in bottlenecks or strategy_focus == OptimizationStrategy.LATENCY_OPTIMIZATION:
            recommendations.append(OptimizationRecommendation(
                strategy=OptimizationStrategy.LATENCY_OPTIMIZATION,
                description="Reduce processing latency through caching and algorithm optimization",
                expected_improvement="30-50% reduction in latency",
                implementation_difficulty="medium",
                estimated_impact_percent=40.0,
                resource_requirements={"development_time": "2-4 days"},
                implementation_steps=[
                    "Implement result caching",
                    "Optimize hot paths",
                    "Use faster algorithms",
                    "Add connection pooling"
                ],
                confidence_score=0.8,
                priority_score=0.85
            ))
        
        # Sort by priority
        recommendations.sort(key=lambda x: x.priority_score, reverse=True)
        
        return recommendations
    
    def _calculate_optimization_score(
        self,
        metrics: PerformanceMetrics,
        bottlenecks: List[str]
    ) -> float:
        """Calculate overall optimization score (0-100)."""
        
        # Start with perfect score
        score = 100.0
        
        # Deduct points for each bottleneck
        if "High CPU usage" in bottlenecks:
            score -= min(20, metrics.cpu_usage_percent - self._cpu_baseline)
        
        if "High memory usage" in bottlenecks:
            memory_penalty = (metrics.memory_usage_mb - self._memory_baseline) / self._memory_baseline * 100
            score -= min(25, memory_penalty)
        
        if "Low throughput" in bottlenecks:
            throughput_penalty = (self._throughput_baseline - metrics.throughput_samples_per_second) / self._throughput_baseline * 100
            score -= min(20, throughput_penalty)
        
        if "High latency" in bottlenecks:
            latency_penalty = (metrics.latency_ms - self._latency_baseline) / self._latency_baseline * 100
            score -= min(20, latency_penalty)
        
        if "Memory pressure" in bottlenecks:
            score -= 15
        
        return max(0.0, score)
    
    def _estimate_improvements(
        self,
        recommendations: List[OptimizationRecommendation]
    ) -> Dict[str, float]:
        """Estimate performance improvements from recommendations."""
        
        improvements = {
            "cpu_reduction_percent": 0.0,
            "memory_reduction_percent": 0.0,
            "throughput_increase_percent": 0.0,
            "latency_reduction_percent": 0.0
        }
        
        for rec in recommendations:
            if rec.strategy == OptimizationStrategy.CPU_OPTIMIZATION:
                improvements["cpu_reduction_percent"] += rec.estimated_impact_percent * 0.3
            elif rec.strategy == OptimizationStrategy.MEMORY_OPTIMIZATION:
                improvements["memory_reduction_percent"] += rec.estimated_impact_percent * 0.4
            elif rec.strategy == OptimizationStrategy.THROUGHPUT_OPTIMIZATION:
                improvements["throughput_increase_percent"] += rec.estimated_impact_percent
            elif rec.strategy == OptimizationStrategy.LATENCY_OPTIMIZATION:
                improvements["latency_reduction_percent"] += rec.estimated_impact_percent * 0.5
        
        return improvements
    
    def _calculate_growth_rate(self, values: List[float]) -> float:
        """Calculate growth rate of a metric."""
        if len(values) < 2:
            return 0.0
        
        # Simple linear growth rate
        return (values[-1] - values[0]) / len(values)
    
    def _analyze_trends(self, history: List[PerformanceMetrics]) -> Dict[str, str]:
        """Analyze performance trends."""
        if len(history) < 10:
            return {"status": "insufficient_data"}
        
        trends = {}
        
        # CPU trend
        cpu_values = [m.cpu_usage_percent for m in history]
        cpu_trend = self._calculate_trend(cpu_values)
        trends["cpu"] = cpu_trend
        
        # Memory trend
        memory_values = [m.memory_usage_mb for m in history]
        memory_trend = self._calculate_trend(memory_values)
        trends["memory"] = memory_trend
        
        # Throughput trend
        throughput_values = [m.throughput_samples_per_second for m in history if m.throughput_samples_per_second > 0]
        if throughput_values:
            throughput_trend = self._calculate_trend(throughput_values)
            trends["throughput"] = throughput_trend
        
        return trends
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a list of values."""
        if len(values) < 5:
            return "stable"
        
        # Simple trend analysis
        first_half = np.mean(values[:len(values)//2])
        second_half = np.mean(values[len(values)//2:])
        
        change_percent = ((second_half - first_half) / first_half) * 100 if first_half > 0 else 0
        
        if change_percent > 10:
            return "increasing"
        elif change_percent < -10:
            return "decreasing"
        else:
            return "stable"


# Global instance management
_performance_optimizer: Optional[PerformanceOptimizer] = None


def get_performance_optimizer() -> PerformanceOptimizer:
    """Get the global performance optimizer instance."""
    global _performance_optimizer
    
    if _performance_optimizer is None:
        _performance_optimizer = PerformanceOptimizer()
    
    return _performance_optimizer


def initialize_performance_optimizer(
    profiling_enabled: bool = True,
    optimization_threshold: float = 0.7
) -> PerformanceOptimizer:
    """Initialize the global performance optimizer instance."""
    global _performance_optimizer
    
    _performance_optimizer = PerformanceOptimizer(
        profiling_enabled=profiling_enabled,
        optimization_threshold=optimization_threshold
    )
    
    return _performance_optimizer