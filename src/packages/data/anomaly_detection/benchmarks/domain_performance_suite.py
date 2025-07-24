#!/usr/bin/env python3
"""Domain-aware performance benchmarking suite for anomaly detection."""

import asyncio
import time
import statistics
import json
import sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Callable
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Benchmark result data structure."""
    name: str
    domain: str
    operation: str
    execution_time_ms: float
    memory_usage_mb: float
    throughput_ops_per_sec: float
    data_size: int
    timestamp: str
    metadata: Dict[str, Any]
    error: Optional[str] = None


class DomainPerformanceBenchmark:
    """Performance benchmarking across domain boundaries."""
    
    def __init__(self):
        """Initialize benchmark suite."""
        self.results: List[BenchmarkResult] = []
        self.warmup_iterations = 3
        self.benchmark_iterations = 10
        
    def _measure_memory(self) -> float:
        """Measure current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    async def run_benchmark(
        self,
        name: str,
        domain: str,
        operation: str,
        benchmark_func: Callable[[], Any],
        data_size: int = 1000,
        metadata: Optional[Dict[str, Any]] = None
    ) -> BenchmarkResult:
        """Run a single benchmark."""
        logger.info(f"Running benchmark: {name} ({domain})")
        
        # Warmup iterations
        for _ in range(self.warmup_iterations):
            try:
                await asyncio.get_event_loop().run_in_executor(None, benchmark_func)
            except Exception:
                pass  # Ignore warmup errors
        
        # Actual benchmark iterations
        execution_times = []
        memory_before = self._measure_memory()
        
        start_time = time.time()
        
        for i in range(self.benchmark_iterations):
            iteration_start = time.time()
            
            try:
                await asyncio.get_event_loop().run_in_executor(None, benchmark_func)
                iteration_time = (time.time() - iteration_start) * 1000
                execution_times.append(iteration_time)
                
            except Exception as e:
                return BenchmarkResult(
                    name=name,
                    domain=domain,
                    operation=operation,
                    execution_time_ms=0.0,
                    memory_usage_mb=0.0,
                    throughput_ops_per_sec=0.0,
                    data_size=data_size,
                    timestamp=datetime.utcnow().isoformat(),
                    metadata=metadata or {},
                    error=str(e)
                )
        
        total_time = time.time() - start_time
        memory_after = self._measure_memory()
        
        # Calculate metrics
        avg_execution_time = statistics.mean(execution_times)
        memory_usage = max(0, memory_after - memory_before)
        throughput = (self.benchmark_iterations * data_size) / total_time if total_time > 0 else 0
        
        result = BenchmarkResult(
            name=name,
            domain=domain,
            operation=operation,
            execution_time_ms=avg_execution_time,
            memory_usage_mb=memory_usage,
            throughput_ops_per_sec=throughput,
            data_size=data_size,
            timestamp=datetime.utcnow().isoformat(),
            metadata={
                **(metadata or {}),
                "iterations": self.benchmark_iterations,
                "min_time_ms": min(execution_times),
                "max_time_ms": max(execution_times),
                "std_dev_ms": statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
                "total_time_sec": total_time
            }
        )
        
        self.results.append(result)
        logger.info(f"Benchmark completed: {name} - {avg_execution_time:.2f}ms avg")
        
        return result
    
    async def benchmark_ai_ml_domain(self) -> List[BenchmarkResult]:
        """Benchmark AI/ML domain performance."""
        results = []
        
        # Test data sizes
        data_sizes = [100, 1000, 5000]
        
        for data_size in data_sizes:
            # Generate test data
            test_data = np.random.rand(data_size, 10)
            
            # Isolation Forest benchmark
            def isolation_forest_benchmark():
                from sklearn.ensemble import IsolationForest
                detector = IsolationForest(random_state=42, n_estimators=100)
                detector.fit(test_data)
                predictions = detector.predict(test_data)
                return predictions
            
            result = await self.run_benchmark(
                name=f"isolation_forest_{data_size}",
                domain="ai_ml",
                operation="anomaly_detection",
                benchmark_func=isolation_forest_benchmark,
                data_size=data_size,
                metadata={
                    "algorithm": "isolation_forest",
                    "n_estimators": 100,
                    "features": 10
                }
            )
            results.append(result)
            
            # Local Outlier Factor benchmark
            def lof_benchmark():
                from sklearn.neighbors import LocalOutlierFactor
                detector = LocalOutlierFactor(n_neighbors=20)
                predictions = detector.fit_predict(test_data)
                return predictions
            
            result = await self.run_benchmark(
                name=f"local_outlier_factor_{data_size}",
                domain="ai_ml",
                operation="anomaly_detection",
                benchmark_func=lof_benchmark,
                data_size=data_size,
                metadata={
                    "algorithm": "local_outlier_factor",
                    "n_neighbors": 20,
                    "features": 10
                }
            )
            results.append(result)
        
        return results
    
    async def benchmark_data_processing_domain(self) -> List[BenchmarkResult]:
        """Benchmark data processing domain performance."""
        results = []
        
        data_sizes = [1000, 10000, 50000]
        
        for data_size in data_sizes:
            # DataFrame operations benchmark
            def dataframe_operations_benchmark():
                df = pd.DataFrame(
                    np.random.rand(data_size, 5),
                    columns=[f'feature_{i}' for i in range(5)]
                )
                
                # Common data processing operations
                df_normalized = (df - df.mean()) / df.std()
                df_filtered = df_normalized[df_normalized['feature_0'] > 0]
                result = df_filtered.describe()
                return result
            
            result = await self.run_benchmark(
                name=f"dataframe_operations_{data_size}",
                domain="data_processing",
                operation="data_transformation",
                benchmark_func=dataframe_operations_benchmark,
                data_size=data_size,
                metadata={
                    "operation_type": "normalize_filter_describe",
                    "columns": 5
                }
            )
            results.append(result)
            
            # NumPy operations benchmark
            def numpy_operations_benchmark():
                data = np.random.rand(data_size, 5)
                
                # Common NumPy operations
                normalized = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
                filtered = normalized[normalized[:, 0] > 0]
                result = np.cov(filtered.T)
                return result
            
            result = await self.run_benchmark(
                name=f"numpy_operations_{data_size}",
                domain="data_processing",
                operation="numerical_computation",
                benchmark_func=numpy_operations_benchmark,
                data_size=data_size,
                metadata={
                    "operation_type": "normalize_filter_covariance",
                    "dimensions": 5
                }
            )
            results.append(result)
        
        return results
    
    async def benchmark_application_domain(self) -> List[BenchmarkResult]:
        """Benchmark application domain performance."""
        results = []
        
        # Detection service benchmark
        def detection_service_benchmark():
            from anomaly_detection.domain.services.detection_service import DetectionService
            
            service = DetectionService()
            test_data = np.random.rand(1000, 4)
            
            result = service.detect_anomalies(
                data=test_data,
                algorithm="iforest",
                contamination=0.1
            )
            return result
        
        result = await self.run_benchmark(
            name="detection_service_iforest",
            domain="application",
            operation="service_detection",
            benchmark_func=detection_service_benchmark,
            data_size=1000,
            metadata={
                "service": "DetectionService",
                "algorithm": "iforest",
                "contamination": 0.1
            }
        )
        results.append(result)
        
        # Ensemble service benchmark
        def ensemble_service_benchmark():
            from anomaly_detection.domain.services.ensemble_service import EnsembleService
            
            service = EnsembleService()
            test_data = np.random.rand(500, 3)
            
            result = service.detect_with_ensemble(
                data=test_data,
                algorithms=["iforest", "lof"],
                combination_method="majority",
                contamination=0.1
            )
            return result
        
        result = await self.run_benchmark(
            name="ensemble_service_majority",
            domain="application",
            operation="ensemble_detection",
            benchmark_func=ensemble_service_benchmark,
            data_size=500,
            metadata={
                "service": "EnsembleService",
                "algorithms": ["iforest", "lof"],
                "combination_method": "majority"
            }
        )
        results.append(result)
        
        return results
    
    async def benchmark_infrastructure_domain(self) -> List[BenchmarkResult]:
        """Benchmark infrastructure domain performance."""
        results = []
        
        # Configuration loading benchmark
        def config_loading_benchmark():
            import os
            import json
            import tempfile
            
            # Simulate configuration loading
            config_data = {
                "database_url": "postgresql://test:test@localhost:5432/test",
                "redis_url": "redis://localhost:6379/0",
                "model_directory": "/app/models",
                "algorithms": ["iforest", "lof", "ocsvm"],
                "default_contamination": 0.1
            }
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(config_data, f)
                config_file = f.name
            
            try:
                with open(config_file, 'r') as f:
                    loaded_config = json.load(f)
                return loaded_config
            finally:
                os.unlink(config_file)
        
        result = await self.run_benchmark(
            name="config_loading",
            domain="infrastructure",
            operation="configuration_management",
            benchmark_func=config_loading_benchmark,
            data_size=1,
            metadata={
                "operation_type": "json_config_load",
                "config_keys": 5
            }
        )
        results.append(result)
        
        # Logging performance benchmark
        def logging_benchmark():
            import logging
            import tempfile
            
            # Setup temporary log file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
                log_file = f.name
            
            try:
                # Configure logger
                logger = logging.getLogger('benchmark_logger')
                logger.setLevel(logging.INFO)
                handler = logging.FileHandler(log_file)
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                logger.addHandler(handler)
                
                # Log messages
                for i in range(100):
                    logger.info(f"Benchmark log message {i} with some data")
                
                logger.removeHandler(handler)
                handler.close()
                
                return i + 1
            finally:
                try:
                    import os
                    os.unlink(log_file)
                except:
                    pass
        
        result = await self.run_benchmark(
            name="logging_performance",
            domain="infrastructure", 
            operation="logging",
            benchmark_func=logging_benchmark,
            data_size=100,
            metadata={
                "operation_type": "file_logging",
                "messages": 100
            }
        )
        results.append(result)
        
        return results
    
    async def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive benchmarks across all domains."""
        logger.info("Starting comprehensive domain performance benchmark")
        
        start_time = time.time()
        
        # Run benchmarks for each domain
        ai_ml_results = await self.benchmark_ai_ml_domain()
        data_processing_results = await self.benchmark_data_processing_domain() 
        application_results = await self.benchmark_application_domain()
        infrastructure_results = await self.benchmark_infrastructure_domain()
        
        total_time = time.time() - start_time
        
        # Aggregate results
        all_results = (
            ai_ml_results + 
            data_processing_results + 
            application_results + 
            infrastructure_results
        )
        
        # Calculate summary statistics
        summary = {
            "benchmark_info": {
                "total_benchmarks": len(all_results),
                "total_time_sec": total_time,
                "timestamp": datetime.utcnow().isoformat(),
                "warmup_iterations": self.warmup_iterations,
                "benchmark_iterations": self.benchmark_iterations
            },
            "domain_summary": {
                "ai_ml": {
                    "count": len(ai_ml_results),
                    "avg_execution_time_ms": statistics.mean([r.execution_time_ms for r in ai_ml_results if not r.error]),
                    "avg_throughput": statistics.mean([r.throughput_ops_per_sec for r in ai_ml_results if not r.error])
                },
                "data_processing": {
                    "count": len(data_processing_results),
                    "avg_execution_time_ms": statistics.mean([r.execution_time_ms for r in data_processing_results if not r.error]),
                    "avg_throughput": statistics.mean([r.throughput_ops_per_sec for r in data_processing_results if not r.error])
                },
                "application": {
                    "count": len(application_results),
                    "avg_execution_time_ms": statistics.mean([r.execution_time_ms for r in application_results if not r.error]),
                    "avg_throughput": statistics.mean([r.throughput_ops_per_sec for r in application_results if not r.error])
                },
                "infrastructure": {
                    "count": len(infrastructure_results),
                    "avg_execution_time_ms": statistics.mean([r.execution_time_ms for r in infrastructure_results if not r.error]),
                    "avg_throughput": statistics.mean([r.throughput_ops_per_sec for r in infrastructure_results if not r.error])
                }
            },
            "results": [asdict(result) for result in all_results]
        }
        
        logger.info(f"Comprehensive benchmark completed in {total_time:.2f}s")
        
        return summary
    
    def save_results(self, filename: str) -> None:
        """Save benchmark results to file."""
        results_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "results": [asdict(result) for result in self.results]
        }
        
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Benchmark results saved to {filename}")


async def main():
    """Main benchmark execution."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    benchmark = DomainPerformanceBenchmark()
    
    try:
        results = await benchmark.run_comprehensive_benchmark()
        
        # Save results
        output_file = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        benchmark.save_results(output_file)
        
        # Print summary
        print("\n" + "="*60)
        print("DOMAIN PERFORMANCE BENCHMARK SUMMARY")
        print("="*60)
        
        print(f"Total benchmarks: {results['benchmark_info']['total_benchmarks']}")
        print(f"Total time: {results['benchmark_info']['total_time_sec']:.2f}s")
        print()
        
        for domain, stats in results['domain_summary'].items():
            print(f"{domain.upper()} Domain:")
            print(f"  Benchmarks: {stats['count']}")
            print(f"  Avg execution time: {stats['avg_execution_time_ms']:.2f}ms")
            print(f"  Avg throughput: {stats['avg_throughput']:.2f} ops/sec")
            print()
        
        print(f"Detailed results saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())