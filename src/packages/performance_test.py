"""Performance testing suite for optimized hexagonal architecture."""

import asyncio
import time
from typing import Dict, Any
import logging

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'shared', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'ai', 'machine_learning', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'data', 'data_quality', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'interfaces', 'src'))

# Import performance utilities directly
import asyncio
import time
from typing import Any, Dict, Optional, Callable, TypeVar, ParamSpec
from functools import wraps
from dataclasses import dataclass
from contextlib import asynccontextmanager

P = ParamSpec('P')
T = TypeVar('T')

@dataclass
class PerformanceMetrics:
    """Performance metrics for operations."""
    operation_name: str
    execution_time: float
    memory_usage: Optional[int] = None
    success: bool = True
    error_message: Optional[str] = None

class PerformanceTracker:
    """Tracks performance metrics across operations."""
    
    def __init__(self):
        self._metrics: Dict[str, list[PerformanceMetrics]] = {}
    
    def record_metric(self, metric: PerformanceMetrics) -> None:
        """Record a performance metric."""
        if metric.operation_name not in self._metrics:
            self._metrics[metric.operation_name] = []
        self._metrics[metric.operation_name].append(metric)
    
    def get_metrics(self, operation_name: str) -> list[PerformanceMetrics]:
        """Get metrics for a specific operation."""
        return self._metrics.get(operation_name, [])
    
    def get_average_time(self, operation_name: str) -> float:
        """Get average execution time for an operation."""
        metrics = self._metrics.get(operation_name, [])
        if not metrics:
            return 0.0
        return sum(m.execution_time for m in metrics) / len(metrics)

# Global performance tracker
performance_tracker = PerformanceTracker()
from machine_learning.infrastructure.adapters.optimized_file_adapter import (
    OptimizedFileBasedDataIngestion,
    OptimizedFileBasedDataProcessing,
    OptimizedFileBasedDataStorage
)
from data_quality.infrastructure.adapters.optimized_adapters import (
    OptimizedDataProfiling,
    OptimizedDataValidation,
    OptimizedStatisticalAnalysis
)
from data_quality.domain.entities.data_profiling_request import DataProfilingRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceBenchmark:
    """Performance benchmark suite for hexagonal architecture optimization."""
    
    def __init__(self):
        self.results: Dict[str, Any] = {}
    
    async def run_ml_performance_tests(self):
        """Run machine learning performance tests."""
        logger.info("Starting ML performance tests...")
        
        # Initialize optimized adapters
        ingestion = OptimizedFileBasedDataIngestion()
        processing = OptimizedFileBasedDataProcessing()
        storage = OptimizedFileBasedDataStorage()
        
        # Test data ingestion performance
        start_time = time.time()
        training_data = await ingestion.ingest_data("performance_test")
        ingestion_time = time.time() - start_time
        
        # Test batch ingestion
        sources = [f"batch_source_{i}" for i in range(10)]
        start_time = time.time()
        batch_results = await ingestion.batch_ingest_data(sources)
        batch_ingestion_time = time.time() - start_time
        
        # Test data processing
        start_time = time.time()
        processed_data = await processing.process_data(training_data)
        processing_time = time.time() - start_time
        
        # Test data validation
        start_time = time.time()
        is_valid = await processing.validate_data(processed_data)
        validation_time = time.time() - start_time
        
        # Test data storage
        start_time = time.time()
        stored = await storage.store_data(processed_data, "performance_test")
        storage_time = time.time() - start_time
        
        # Test data retrieval
        start_time = time.time()
        retrieved_data = await storage.retrieve_data("performance_test")
        retrieval_time = time.time() - start_time
        
        self.results["ml_performance"] = {
            "ingestion_time": ingestion_time,
            "batch_ingestion_time": batch_ingestion_time,
            "processing_time": processing_time,
            "validation_time": validation_time,
            "storage_time": storage_time,
            "retrieval_time": retrieval_time,
            "total_operations_time": sum([
                ingestion_time, batch_ingestion_time, processing_time,
                validation_time, storage_time, retrieval_time
            ])
        }
        
        logger.info(f"ML performance tests completed in {self.results['ml_performance']['total_operations_time']:.3f}s")
    
    async def run_data_quality_performance_tests(self):
        """Run data quality performance tests."""
        logger.info("Starting Data Quality performance tests...")
        
        # Initialize optimized adapters
        profiling = OptimizedDataProfiling()
        validation = OptimizedDataValidation()
        analysis = OptimizedStatisticalAnalysis()
        
        # Test data profiling performance
        request = DataProfilingRequest(
            data_source="performance_test_dq",
            timestamp="2024-01-01T00:00:00Z"
        )
        
        start_time = time.time()
        profile = await profiling.create_data_profile(request)
        profiling_time = time.time() - start_time
        
        # Test cached profiling (should be faster)
        start_time = time.time()
        cached_profile = await profiling.create_data_profile(request)
        cached_profiling_time = time.time() - start_time
        
        # Test data validation performance
        from data_quality.domain.entities.data_quality_rule import DataQualityRule
        
        rules = [
            DataQualityRule(rule_name=f"test_rule_{i}", description=f"Test rule {i}")
            for i in range(20)
        ]
        
        start_time = time.time()
        validation_results = await validation.validate_data("performance_test_dq", rules)
        validation_time = time.time() - start_time
        
        # Test statistical analysis performance
        start_time = time.time()
        stats_report = await analysis.analyze_data("performance_test_dq")
        analysis_time = time.time() - start_time
        
        # Test cached analysis (should be faster)
        start_time = time.time()
        cached_stats = await analysis.analyze_data("performance_test_dq")
        cached_analysis_time = time.time() - start_time
        
        self.results["data_quality_performance"] = {
            "profiling_time": profiling_time,
            "cached_profiling_time": cached_profiling_time,
            "validation_time": validation_time,
            "analysis_time": analysis_time,
            "cached_analysis_time": cached_analysis_time,
            "total_operations_time": sum([
                profiling_time, validation_time, analysis_time
            ]),
            "cache_improvement": {
                "profiling_speedup": profiling_time / cached_profiling_time if cached_profiling_time > 0 else 0,
                "analysis_speedup": analysis_time / cached_analysis_time if cached_analysis_time > 0 else 0
            }
        }
        
        logger.info(f"Data Quality performance tests completed in {self.results['data_quality_performance']['total_operations_time']:.3f}s")
    
    async def run_concurrent_load_test(self):
        """Run concurrent load test to measure scalability."""
        logger.info("Starting concurrent load test...")
        
        # Test concurrent ML operations
        ml_ingestion = OptimizedFileBasedDataIngestion()
        
        async def ml_load_test():
            tasks = []
            for i in range(50):  # 50 concurrent ingestion operations
                tasks.append(ml_ingestion.ingest_data(f"load_test_{i}"))
            
            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()
            
            successful = sum(1 for r in results if not isinstance(r, Exception))
            return end_time - start_time, successful, len(results)
        
        # Test concurrent data quality operations
        dq_profiling = OptimizedDataProfiling()
        
        async def dq_load_test():
            tasks = []
            for i in range(30):  # 30 concurrent profiling operations
                request = DataProfilingRequest(
                    data_source=f"load_test_dq_{i}",
                    timestamp="2024-01-01T00:00:00Z"
                )
                tasks.append(dq_profiling.create_data_profile(request))
            
            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()
            
            successful = sum(1 for r in results if not isinstance(r, Exception))
            return end_time - start_time, successful, len(results)
        
        # Run load tests
        ml_time, ml_successful, ml_total = await ml_load_test()
        dq_time, dq_successful, dq_total = await dq_load_test()
        
        self.results["concurrent_load_test"] = {
            "ml_concurrent_time": ml_time,
            "ml_success_rate": ml_successful / ml_total,
            "ml_throughput": ml_successful / ml_time,
            "dq_concurrent_time": dq_time,
            "dq_success_rate": dq_successful / dq_total,
            "dq_throughput": dq_successful / dq_time
        }
        
        logger.info(f"Concurrent load test completed - ML: {ml_successful}/{ml_total} ops in {ml_time:.3f}s, DQ: {dq_successful}/{dq_total} ops in {dq_time:.3f}s")
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        # Get performance metrics from tracker
        all_metrics = {}
        for operation_name in ["data_ingestion", "data_processing", "data_profiling", "data_validation", "statistical_analysis"]:
            metrics = performance_tracker.get_metrics(operation_name)
            if metrics:
                avg_time = performance_tracker.get_average_time(operation_name)
                all_metrics[operation_name] = {
                    "average_time": avg_time,
                    "total_calls": len(metrics),
                    "success_rate": sum(1 for m in metrics if m.success) / len(metrics)
                }
        
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_results": self.results,
            "operation_metrics": all_metrics,
            "summary": {
                "total_test_time": sum(
                    result.get("total_operations_time", 0) 
                    for result in self.results.values() 
                    if isinstance(result, dict)
                ),
                "optimizations_applied": [
                    "Caching with TTL",
                    "Batch processing",
                    "Parallel execution",
                    "Connection pooling",
                    "Memory-efficient chunking",
                    "Sample-based analysis for large datasets",
                    "Vectorized operations",
                    "Concurrent operation limiting"
                ]
            }
        }
        
        return report

async def main():
    """Run comprehensive performance benchmark."""
    benchmark = PerformanceBenchmark()
    
    # Run all performance tests
    await benchmark.run_ml_performance_tests()
    await benchmark.run_data_quality_performance_tests()
    await benchmark.run_concurrent_load_test()
    
    # Generate and display report
    report = benchmark.generate_performance_report()
    
    print("\n" + "="*80)
    print("HEXAGONAL ARCHITECTURE PERFORMANCE BENCHMARK REPORT")
    print("="*80)
    
    print(f"\nTest completed at: {report['timestamp']}")
    print(f"Total test execution time: {report['summary']['total_test_time']:.3f} seconds")
    
    print("\n📊 MACHINE LEARNING PERFORMANCE:")
    ml_perf = report['test_results']['ml_performance']
    print(f"  • Data ingestion: {ml_perf['ingestion_time']:.3f}s")
    print(f"  • Batch ingestion (10 sources): {ml_perf['batch_ingestion_time']:.3f}s")
    print(f"  • Data processing: {ml_perf['processing_time']:.3f}s")
    print(f"  • Data validation: {ml_perf['validation_time']:.3f}s")
    print(f"  • Data storage: {ml_perf['storage_time']:.3f}s")
    print(f"  • Data retrieval: {ml_perf['retrieval_time']:.3f}s")
    
    print("\n📊 DATA QUALITY PERFORMANCE:")
    dq_perf = report['test_results']['data_quality_performance']
    print(f"  • Data profiling: {dq_perf['profiling_time']:.3f}s")
    print(f"  • Cached profiling: {dq_perf['cached_profiling_time']:.3f}s (speedup: {dq_perf['cache_improvement']['profiling_speedup']:.1f}x)")
    print(f"  • Data validation (20 rules): {dq_perf['validation_time']:.3f}s")
    print(f"  • Statistical analysis: {dq_perf['analysis_time']:.3f}s")
    print(f"  • Cached analysis: {dq_perf['cached_analysis_time']:.3f}s (speedup: {dq_perf['cache_improvement']['analysis_speedup']:.1f}x)")
    
    print("\n📊 CONCURRENT LOAD TEST:")
    load_test = report['test_results']['concurrent_load_test']
    print(f"  • ML concurrent ops: {load_test['ml_success_rate']:.1%} success rate, {load_test['ml_throughput']:.1f} ops/sec")
    print(f"  • DQ concurrent ops: {load_test['dq_success_rate']:.1%} success rate, {load_test['dq_throughput']:.1f} ops/sec")
    
    print("\n🚀 OPTIMIZATIONS APPLIED:")
    for optimization in report['summary']['optimizations_applied']:
        print(f"  • {optimization}")
    
    print("\n📈 OPERATION METRICS:")
    for op_name, metrics in report['operation_metrics'].items():
        print(f"  • {op_name}: {metrics['average_time']:.3f}s avg, {metrics['total_calls']} calls, {metrics['success_rate']:.1%} success")
    
    print("\n" + "="*80)
    print("Performance optimization completed successfully!")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(main())