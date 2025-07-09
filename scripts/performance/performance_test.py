#!/usr/bin/env python3
"""
Performance testing script for Pynomaly.
Tests CPU, memory, and scalability performance using profiling tools.
"""

import argparse
import asyncio
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np
import pandas as pd
import psutil
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import requests
import threading

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from pynomaly.infrastructure.performance.profiling_hotspots import (
    PerformanceProfiler,
    NumpyPandasOptimizer,
    example_numpy_heavy_function,
    example_pandas_heavy_function
)
from pynomaly.infrastructure.async_tasks.celery_tasks import TaskManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceTestSuite:
    """Comprehensive performance test suite for Pynomaly."""
    
    def __init__(self, output_dir: str = "performance_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.profiler = PerformanceProfiler(str(self.output_dir))
        self.results = []
        
    def run_profiling_tests(self):
        """Run profiling tests with py-spy and scalene."""
        logger.info("Running profiling tests...")
        
        # Test 1: Profile numpy operations
        logger.info("Profiling numpy operations...")
        try:
            numpy_result = self.profiler.profile_with_py_spy(
                example_numpy_heavy_function,
                duration=10
            )
            self.results.append({
                'test': 'numpy_profiling',
                'tool': 'py-spy',
                'result': numpy_result,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Numpy profiling failed: {e}")
            
        # Test 2: Profile pandas operations
        logger.info("Profiling pandas operations...")
        try:
            pandas_result = self.profiler.profile_with_scalene(
                example_pandas_heavy_function,
                duration=10
            )
            self.results.append({
                'test': 'pandas_profiling',
                'tool': 'scalene',
                'result': pandas_result,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Pandas profiling failed: {e}")
            
    def run_optimization_tests(self):
        """Run optimization tests with decorators."""
        logger.info("Running optimization tests...")
        
        # Test without optimization
        start_time = time.time()
        result_unoptimized = example_numpy_heavy_function()
        unoptimized_time = time.time() - start_time
        
        # Test with optimization
        optimized_function = NumpyPandasOptimizer.optimize_numpy_operations(
            example_numpy_heavy_function
        )
        start_time = time.time()
        result_optimized = optimized_function()
        optimized_time = time.time() - start_time
        
        optimization_result = {
            'test': 'numpy_optimization',
            'unoptimized_time': unoptimized_time,
            'optimized_time': optimized_time,
            'improvement': (unoptimized_time - optimized_time) / unoptimized_time * 100,
            'timestamp': datetime.now().isoformat()
        }
        
        self.results.append(optimization_result)
        logger.info(f"Optimization improvement: {optimization_result['improvement']:.2f}%")
        
    def run_celery_performance_tests(self):
        """Run Celery task performance tests."""
        logger.info("Running Celery performance tests...")
        
        try:
            task_manager = TaskManager()
            
            # Test heavy detection task
            sample_data = np.random.rand(1000, 10)
            start_time = time.time()
            
            task_id = task_manager.submit_heavy_detection(
                dataset=sample_data,
                algorithm='isolation_forest',
                contamination=0.1
            )
            
            # Wait for completion (with timeout)
            timeout = 60  # 1 minute timeout
            elapsed = 0
            while elapsed < timeout:
                status = task_manager.get_task_status(task_id)
                if status.status in ['SUCCESS', 'FAILURE']:
                    break
                time.sleep(1)
                elapsed += 1
                
            end_time = time.time()
            
            celery_result = {
                'test': 'celery_heavy_detection',
                'task_id': task_id,
                'duration': end_time - start_time,
                'status': status.status,
                'dataset_size': sample_data.shape,
                'timestamp': datetime.now().isoformat()
            }
            
            self.results.append(celery_result)
            logger.info(f"Celery task completed in {celery_result['duration']:.2f}s")
            
        except Exception as e:
            logger.error(f"Celery performance test failed: {e}")
            
    def run_concurrent_tests(self, num_workers: int = 4):
        """Run concurrent performance tests."""
        logger.info(f"Running concurrent tests with {num_workers} workers...")
        
        def worker_function(worker_id: int) -> Dict[str, Any]:
            """Worker function for concurrent testing."""
            start_time = time.time()
            
            # Simulate heavy computation
            data = np.random.rand(5000, 50)
            result = np.mean(np.dot(data, data.T))
            
            end_time = time.time()
            
            return {
                'worker_id': worker_id,
                'duration': end_time - start_time,
                'result': float(result),
                'timestamp': datetime.now().isoformat()
            }
            
        # Test ThreadPoolExecutor
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            thread_results = list(executor.map(worker_function, range(num_workers)))
        thread_time = time.time() - start_time
        
        # Test ProcessPoolExecutor
        start_time = time.time()
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            process_results = list(executor.map(worker_function, range(num_workers)))
        process_time = time.time() - start_time
        
        concurrent_result = {
            'test': 'concurrent_performance',
            'num_workers': num_workers,
            'thread_time': thread_time,
            'process_time': process_time,
            'thread_results': thread_results,
            'process_results': process_results,
            'timestamp': datetime.now().isoformat()
        }
        
        self.results.append(concurrent_result)
        logger.info(f"Thread pool: {thread_time:.2f}s, Process pool: {process_time:.2f}s")
        
    def run_memory_tests(self):
        """Run memory usage tests."""
        logger.info("Running memory tests...")
        
        def memory_intensive_function(size: int) -> Dict[str, Any]:
            """Memory intensive function for testing."""
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # Allocate large arrays
            arrays = []
            for i in range(10):
                arrays.append(np.random.rand(size, size))
                
            peak_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # Clean up
            del arrays
            
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            return {
                'size': size,
                'start_memory_mb': start_memory,
                'peak_memory_mb': peak_memory,
                'end_memory_mb': end_memory,
                'memory_used_mb': peak_memory - start_memory
            }
            
        # Test different array sizes
        sizes = [100, 500, 1000, 2000]
        memory_results = []
        
        for size in sizes:
            result = memory_intensive_function(size)
            memory_results.append(result)
            logger.info(f"Size {size}x{size}: {result['memory_used_mb']:.2f} MB used")
            
        memory_test_result = {
            'test': 'memory_usage',
            'results': memory_results,
            'timestamp': datetime.now().isoformat()
        }
        
        self.results.append(memory_test_result)
        
    def run_load_test(self, url: str = "http://localhost:8000", 
                     num_requests: int = 1000, 
                     concurrency: int = 10):
        """Run load test against API endpoint."""
        logger.info(f"Running load test against {url}...")
        
        def send_request(request_id: int) -> Dict[str, Any]:
            """Send single request and measure response time."""
            start_time = time.time()
            try:
                response = requests.get(f"{url}/health", timeout=10)
                end_time = time.time()
                
                return {
                    'request_id': request_id,
                    'duration': end_time - start_time,
                    'status_code': response.status_code,
                    'success': response.status_code == 200,
                    'timestamp': datetime.now().isoformat()
                }
            except Exception as e:
                end_time = time.time()
                return {
                    'request_id': request_id,
                    'duration': end_time - start_time,
                    'status_code': None,
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                
        # Run load test
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            request_results = list(executor.map(send_request, range(num_requests)))
        total_time = time.time() - start_time
        
        # Calculate statistics
        successful_requests = [r for r in request_results if r['success']]
        failed_requests = [r for r in request_results if not r['success']]
        
        if successful_requests:
            response_times = [r['duration'] for r in successful_requests]
            avg_response_time = np.mean(response_times)
            p95_response_time = np.percentile(response_times, 95)
            p99_response_time = np.percentile(response_times, 99)
        else:
            avg_response_time = p95_response_time = p99_response_time = 0
            
        requests_per_second = num_requests / total_time
        success_rate = len(successful_requests) / num_requests * 100
        
        load_test_result = {
            'test': 'load_test',
            'url': url,
            'num_requests': num_requests,
            'concurrency': concurrency,
            'total_time': total_time,
            'requests_per_second': requests_per_second,
            'success_rate': success_rate,
            'avg_response_time': avg_response_time,
            'p95_response_time': p95_response_time,
            'p99_response_time': p99_response_time,
            'successful_requests': len(successful_requests),
            'failed_requests': len(failed_requests),
            'timestamp': datetime.now().isoformat()
        }
        
        self.results.append(load_test_result)
        logger.info(f"Load test completed: {requests_per_second:.2f} RPS, "
                   f"{success_rate:.2f}% success rate")
        
    def run_scalability_test(self, worker_counts: List[int] = [1, 2, 4, 8]):
        """Run scalability test with different worker counts."""
        logger.info("Running scalability test...")
        
        def benchmark_function(data_size: int) -> float:
            """Benchmark function for scalability testing."""
            data = np.random.rand(data_size, data_size)
            return np.mean(np.dot(data, data.T))
            
        scalability_results = []
        data_size = 1000
        
        for worker_count in worker_counts:
            start_time = time.time()
            
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                # Submit multiple tasks
                futures = [executor.submit(benchmark_function, data_size) 
                          for _ in range(worker_count * 2)]
                
                # Wait for completion
                results = [future.result() for future in futures]
                
            end_time = time.time()
            
            scalability_result = {
                'worker_count': worker_count,
                'duration': end_time - start_time,
                'tasks_completed': len(results),
                'throughput': len(results) / (end_time - start_time),
                'timestamp': datetime.now().isoformat()
            }
            
            scalability_results.append(scalability_result)
            logger.info(f"Workers {worker_count}: {scalability_result['throughput']:.2f} tasks/sec")
            
        scalability_test_result = {
            'test': 'scalability',
            'data_size': data_size,
            'results': scalability_results,
            'timestamp': datetime.now().isoformat()
        }
        
        self.results.append(scalability_test_result)
        
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
                'memory_total': psutil.virtual_memory().total,
                'memory_available': psutil.virtual_memory().available,
                'platform': sys.platform,
                'python_version': sys.version
            },
            'tests_run': len(self.results),
            'results': self.results
        }
        
        # Save report
        report_path = self.output_dir / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Performance report saved to {report_path}")
        return report
        
    def run_all_tests(self, skip_profiling: bool = False, 
                     skip_load_test: bool = False,
                     api_url: str = "http://localhost:8000"):
        """Run all performance tests."""
        logger.info("Starting comprehensive performance test suite...")
        
        # Run profiling tests (can be slow)
        if not skip_profiling:
            self.run_profiling_tests()
            
        # Run optimization tests
        self.run_optimization_tests()
        
        # Run Celery performance tests
        self.run_celery_performance_tests()
        
        # Run concurrent tests
        self.run_concurrent_tests()
        
        # Run memory tests
        self.run_memory_tests()
        
        # Run scalability tests
        self.run_scalability_test()
        
        # Run load test (requires running API server)
        if not skip_load_test:
            try:
                self.run_load_test(api_url)
            except Exception as e:
                logger.warning(f"Load test failed (API server may not be running): {e}")
                
        # Generate final report
        report = self.generate_report()
        
        logger.info("Performance test suite completed!")
        return report


def main():
    """Main entry point for performance testing."""
    parser = argparse.ArgumentParser(description="Pynomaly Performance Test Suite")
    
    parser.add_argument(
        '--output-dir', 
        default='performance_results',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--skip-profiling',
        action='store_true',
        help='Skip profiling tests (py-spy and scalene)'
    )
    
    parser.add_argument(
        '--skip-load-test',
        action='store_true',
        help='Skip load test (requires running API server)'
    )
    
    parser.add_argument(
        '--api-url',
        default='http://localhost:8000',
        help='API URL for load testing'
    )
    
    parser.add_argument(
        '--test-type',
        choices=['all', 'profiling', 'optimization', 'celery', 'concurrent', 'memory', 'load', 'scalability'],
        default='all',
        help='Type of test to run'
    )
    
    args = parser.parse_args()
    
    # Create test suite
    test_suite = PerformanceTestSuite(args.output_dir)
    
    # Run specific test or all tests
    if args.test_type == 'all':
        test_suite.run_all_tests(args.skip_profiling, args.skip_load_test, args.api_url)
    elif args.test_type == 'profiling':
        test_suite.run_profiling_tests()
    elif args.test_type == 'optimization':
        test_suite.run_optimization_tests()
    elif args.test_type == 'celery':
        test_suite.run_celery_performance_tests()
    elif args.test_type == 'concurrent':
        test_suite.run_concurrent_tests()
    elif args.test_type == 'memory':
        test_suite.run_memory_tests()
    elif args.test_type == 'load':
        test_suite.run_load_test(args.api_url)
    elif args.test_type == 'scalability':
        test_suite.run_scalability_test()
        
    # Generate report for specific tests
    if args.test_type != 'all':
        test_suite.generate_report()


if __name__ == '__main__':
    main()
