#!/usr/bin/env python3
"""
Production Load Testing Framework for Monorepo.
Tests system performance under production-scale loads with comprehensive metrics.
"""

import asyncio
import time
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
import numpy as np
import pandas as pd
import psutil
import json
import sys

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@dataclass
class LoadTestConfig:
    """Configuration for load testing scenarios."""
    test_name: str
    duration_seconds: int = 300  # 5 minutes default
    concurrent_users: int = 10
    requests_per_second: int = 100
    data_size_mb: int = 10
    ramp_up_seconds: int = 30
    warm_up_seconds: int = 60
    memory_limit_mb: int = 2048
    cpu_limit_percent: int = 80
    success_rate_threshold: float = 0.95
    response_time_p95_ms: int = 1000


@dataclass
class LoadTestResult:
    """Results from load testing execution."""
    test_name: str
    success: bool
    duration_seconds: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    requests_per_second: float
    peak_memory_mb: float
    peak_cpu_percent: float
    error_details: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)


class SystemMonitor:
    """Monitor system resources during load testing."""
    
    def __init__(self):
        self.monitoring = False
        self.metrics = []
        self.monitor_thread = None
        
    def start_monitoring(self, interval_seconds: float = 1.0):
        """Start system resource monitoring."""
        self.monitoring = True
        self.metrics = []
        
        def monitor_resources():
            process = psutil.Process()
            
            while self.monitoring:
                try:
                    metric = {
                        'timestamp': time.time(),
                        'cpu_percent': process.cpu_percent(),
                        'memory_mb': process.memory_info().rss / 1024 / 1024,
                        'system_cpu_percent': psutil.cpu_percent(),
                        'system_memory_percent': psutil.virtual_memory().percent,
                        'system_memory_available_mb': psutil.virtual_memory().available / 1024 / 1024,
                        'disk_io': psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else {},
                        'network_io': psutil.net_io_counters()._asdict() if psutil.net_io_counters() else {},
                        'open_files': len(process.open_files()),
                        'connections': len(process.connections())
                    }
                    self.metrics.append(metric)
                    
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    # Handle process termination or permission issues
                    pass
                    
                time.sleep(interval_seconds)
        
        self.monitor_thread = threading.Thread(target=monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return aggregated metrics."""
        self.monitoring = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        if not self.metrics:
            return {}
        
        # Calculate aggregated metrics
        cpu_values = [m['cpu_percent'] for m in self.metrics]
        memory_values = [m['memory_mb'] for m in self.metrics]
        system_cpu_values = [m['system_cpu_percent'] for m in self.metrics]
        system_memory_values = [m['system_memory_percent'] for m in self.metrics]
        
        return {
            'peak_cpu_percent': max(cpu_values),
            'average_cpu_percent': sum(cpu_values) / len(cpu_values),
            'peak_memory_mb': max(memory_values),
            'average_memory_mb': sum(memory_values) / len(memory_values),
            'peak_system_cpu_percent': max(system_cpu_values),
            'peak_system_memory_percent': max(system_memory_values),
            'samples_collected': len(self.metrics),
            'monitoring_duration': self.metrics[-1]['timestamp'] - self.metrics[0]['timestamp']
        }


class DataGenerator:
    """Generate test data for load testing."""
    
    @staticmethod
    def generate_anomaly_dataset(size_mb: int, contamination: float = 0.1) -> pd.DataFrame:
        """Generate synthetic anomaly detection dataset."""
        # Calculate approximate rows for target size
        bytes_per_row = 80  # Estimated for typical anomaly data
        target_rows = int((size_mb * 1024 * 1024) / bytes_per_row)
        
        np.random.seed(42)
        
        # Generate normal data
        n_normal = int(target_rows * (1 - contamination))
        n_features = 10
        
        normal_data = np.random.multivariate_normal(
            mean=np.zeros(n_features),
            cov=np.eye(n_features),
            size=n_normal
        )
        
        # Generate anomalous data
        n_anomalies = target_rows - n_normal
        anomaly_data = np.random.multivariate_normal(
            mean=np.ones(n_features) * 3,
            cov=np.eye(n_features) * 2,
            size=n_anomalies
        )
        
        # Combine data
        X = np.vstack([normal_data, anomaly_data])
        labels = np.hstack([np.ones(n_normal), -np.ones(n_anomalies)])
        
        # Create DataFrame
        columns = [f'feature_{i}' for i in range(n_features)]
        df = pd.DataFrame(X, columns=columns)
        df['label'] = labels
        df['timestamp'] = pd.date_range('2024-01-01', periods=len(df), freq='s')
        
        return df
    
    @staticmethod
    def generate_quality_dataset(size_mb: int, quality_issues: bool = True) -> pd.DataFrame:
        """Generate data quality test dataset."""
        bytes_per_row = 100
        target_rows = int((size_mb * 1024 * 1024) / bytes_per_row)
        
        np.random.seed(42)
        
        data = {
            'id': range(target_rows),
            'numeric_col_1': np.random.randn(target_rows),
            'numeric_col_2': np.random.exponential(2, target_rows),
            'category_col': np.random.choice(['A', 'B', 'C', 'D'], target_rows, p=[0.4, 0.3, 0.2, 0.1]),
            'datetime_col': pd.date_range('2024-01-01', periods=target_rows, freq='min'),
            'boolean_col': np.random.choice([True, False], target_rows, p=[0.7, 0.3])
        }
        
        df = pd.DataFrame(data)
        
        if quality_issues:
            # Introduce quality issues
            missing_indices = np.random.choice(target_rows, int(target_rows * 0.05), replace=False)
            df.loc[missing_indices, 'numeric_col_1'] = np.nan
            
            # Add duplicates
            duplicate_indices = np.random.choice(target_rows // 2, int(target_rows * 0.02), replace=False)
            for idx in duplicate_indices:
                df.loc[target_rows - idx - 1] = df.loc[idx]
                
            # Add outliers
            outlier_indices = np.random.choice(target_rows, int(target_rows * 0.01), replace=False)
            df.loc[outlier_indices, 'numeric_col_2'] += 10 * df['numeric_col_2'].std()
        
        return df


class LoadTester:
    """Main load testing orchestrator."""
    
    def __init__(self):
        self.monitor = SystemMonitor()
        self.data_generator = DataGenerator()
    
    def run_anomaly_detection_load_test(self, config: LoadTestConfig) -> LoadTestResult:
        """Run load test for anomaly detection pipeline."""
        print(f"🚀 Starting Anomaly Detection Load Test: {config.test_name}")
        print(f"   Duration: {config.duration_seconds}s")
        print(f"   Concurrent Users: {config.concurrent_users}")
        print(f"   Data Size: {config.data_size_mb}MB per request")
        
        # Generate test data
        test_data = self.data_generator.generate_anomaly_dataset(config.data_size_mb)
        print(f"   Generated dataset: {len(test_data)} rows, {test_data.memory_usage(deep=True).sum() / 1024 / 1024:.1f}MB")
        
        # Start system monitoring
        self.monitor.start_monitoring()
        
        # Load test execution
        start_time = time.time()
        response_times = []
        successful_requests = 0
        failed_requests = 0
        error_details = []
        
        def detection_worker():
            """Worker function for anomaly detection."""
            nonlocal successful_requests, failed_requests
            
            try:
                worker_start = time.time()
                
                # Mock anomaly detection processing
                # In real implementation, this would call actual detection service
                result = self._mock_anomaly_detection(test_data)
                
                worker_end = time.time()
                response_time = (worker_end - worker_start) * 1000  # Convert to ms
                
                if result['success']:
                    successful_requests += 1
                    response_times.append(response_time)
                else:
                    failed_requests += 1
                    error_details.append(result.get('error', 'Unknown error'))
                    
            except Exception as e:
                failed_requests += 1
                error_details.append(str(e))
        
        # Execute concurrent load test
        with ThreadPoolExecutor(max_workers=config.concurrent_users) as executor:
            futures = []
            
            end_time = start_time + config.duration_seconds
            
            while time.time() < end_time:
                # Submit workers up to concurrent limit
                if len(futures) < config.concurrent_users:
                    future = executor.submit(detection_worker)
                    futures.append(future)
                
                # Remove completed futures
                futures = [f for f in futures if not f.done()]
                
                # Control request rate
                time.sleep(1.0 / config.requests_per_second)
            
            # Wait for remaining requests to complete
            for future in futures:
                future.result(timeout=30)
        
        # Stop monitoring and collect results
        execution_time = time.time() - start_time
        system_metrics = self.monitor.stop_monitoring()
        
        # Calculate performance metrics
        total_requests = successful_requests + failed_requests
        success_rate = successful_requests / total_requests if total_requests > 0 else 0
        
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            p95_response_time = np.percentile(response_times, 95)
            p99_response_time = np.percentile(response_times, 99)
        else:
            avg_response_time = p95_response_time = p99_response_time = 0
        
        requests_per_sec = total_requests / execution_time if execution_time > 0 else 0
        
        # Determine success
        test_success = (
            success_rate >= config.success_rate_threshold and
            p95_response_time <= config.response_time_p95_ms and
            system_metrics.get('peak_memory_mb', 0) <= config.memory_limit_mb and
            system_metrics.get('peak_cpu_percent', 0) <= config.cpu_limit_percent
        )
        
        result = LoadTestResult(
            test_name=config.test_name,
            success=test_success,
            duration_seconds=execution_time,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            average_response_time_ms=avg_response_time,
            p95_response_time_ms=p95_response_time,
            p99_response_time_ms=p99_response_time,
            requests_per_second=requests_per_sec,
            peak_memory_mb=system_metrics.get('peak_memory_mb', 0),
            peak_cpu_percent=system_metrics.get('peak_cpu_percent', 0),
            error_details=error_details[:10],  # Limit error details
            performance_metrics=system_metrics
        )
        
        self._print_test_results(result, config)
        return result
    
    def run_data_quality_load_test(self, config: LoadTestConfig) -> LoadTestResult:
        """Run load test for data quality pipeline."""
        print(f"🔍 Starting Data Quality Load Test: {config.test_name}")
        
        # Generate test data with quality issues
        test_data = self.data_generator.generate_quality_dataset(config.data_size_mb, quality_issues=True)
        print(f"   Generated quality dataset: {len(test_data)} rows")
        
        # Start monitoring
        self.monitor.start_monitoring()
        
        # Load test execution
        start_time = time.time()
        response_times = []
        successful_requests = 0
        failed_requests = 0
        error_details = []
        
        def quality_worker():
            """Worker function for data quality assessment."""
            nonlocal successful_requests, failed_requests
            
            try:
                worker_start = time.time()
                
                # Mock data quality processing
                result = self._mock_quality_assessment(test_data)
                
                worker_end = time.time()
                response_time = (worker_end - worker_start) * 1000
                
                if result['success']:
                    successful_requests += 1
                    response_times.append(response_time)
                else:
                    failed_requests += 1
                    error_details.append(result.get('error', 'Unknown error'))
                    
            except Exception as e:
                failed_requests += 1
                error_details.append(str(e))
        
        # Execute load test
        with ThreadPoolExecutor(max_workers=config.concurrent_users) as executor:
            futures = []
            end_time = start_time + config.duration_seconds
            
            while time.time() < end_time:
                if len(futures) < config.concurrent_users:
                    future = executor.submit(quality_worker)
                    futures.append(future)
                
                futures = [f for f in futures if not f.done()]
                time.sleep(1.0 / config.requests_per_second)
            
            for future in futures:
                future.result(timeout=30)
        
        # Process results
        execution_time = time.time() - start_time
        system_metrics = self.monitor.stop_monitoring()
        
        total_requests = successful_requests + failed_requests
        success_rate = successful_requests / total_requests if total_requests > 0 else 0
        
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            p95_response_time = np.percentile(response_times, 95)
            p99_response_time = np.percentile(response_times, 99)
        else:
            avg_response_time = p95_response_time = p99_response_time = 0
        
        requests_per_sec = total_requests / execution_time if execution_time > 0 else 0
        
        test_success = (
            success_rate >= config.success_rate_threshold and
            p95_response_time <= config.response_time_p95_ms and
            system_metrics.get('peak_memory_mb', 0) <= config.memory_limit_mb
        )
        
        result = LoadTestResult(
            test_name=config.test_name,
            success=test_success,
            duration_seconds=execution_time,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            average_response_time_ms=avg_response_time,
            p95_response_time_ms=p95_response_time,
            p99_response_time_ms=p99_response_time,
            requests_per_second=requests_per_sec,
            peak_memory_mb=system_metrics.get('peak_memory_mb', 0),
            peak_cpu_percent=system_metrics.get('peak_cpu_percent', 0),
            error_details=error_details[:10],
            performance_metrics=system_metrics
        )
        
        self._print_test_results(result, config)
        return result
    
    def _mock_anomaly_detection(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Mock anomaly detection processing."""
        try:
            # Simulate processing time proportional to data size
            processing_time = len(data) * 0.00001  # 0.01ms per row
            time.sleep(processing_time)
            
            # Mock detection results
            contamination = 0.1
            n_anomalies = int(len(data) * contamination)
            anomaly_indices = np.random.choice(len(data), n_anomalies, replace=False)
            
            return {
                'success': True,
                'anomalies_detected': n_anomalies,
                'anomaly_indices': anomaly_indices.tolist(),
                'processing_time': processing_time,
                'confidence_scores': np.random.uniform(0.7, 0.95, len(data)).tolist()
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _mock_quality_assessment(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Mock data quality assessment processing."""
        try:
            # Simulate quality assessment processing
            processing_time = len(data) * 0.00005  # 0.05ms per row
            time.sleep(processing_time)
            
            # Mock quality metrics
            completeness = 1.0 - (data.isnull().sum().sum() / (len(data) * len(data.columns)))
            
            return {
                'success': True,
                'overall_score': completeness * 0.9,  # Mock overall score
                'completeness': completeness,
                'accuracy': 0.92,
                'consistency': 0.88,
                'processing_time': processing_time,
                'issues_found': max(0, int((1 - completeness) * len(data)))
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _print_test_results(self, result: LoadTestResult, config: LoadTestConfig):
        """Print formatted test results."""
        print(f"\n📊 Load Test Results: {result.test_name}")
        print("=" * 50)
        
        # Test outcome
        status_icon = "✅" if result.success else "❌"
        print(f"{status_icon} Test Status: {'PASSED' if result.success else 'FAILED'}")
        
        # Performance metrics
        print(f"\n📈 Performance Metrics:")
        print(f"   Duration: {result.duration_seconds:.1f}s")
        print(f"   Total Requests: {result.total_requests}")
        print(f"   Success Rate: {(result.successful_requests/result.total_requests*100):.1f}%")
        print(f"   Requests/Second: {result.requests_per_second:.1f}")
        print(f"   Avg Response Time: {result.average_response_time_ms:.1f}ms")
        print(f"   P95 Response Time: {result.p95_response_time_ms:.1f}ms")
        print(f"   P99 Response Time: {result.p99_response_time_ms:.1f}ms")
        
        # Resource utilization
        print(f"\n💾 Resource Utilization:")
        print(f"   Peak Memory: {result.peak_memory_mb:.1f}MB")
        print(f"   Peak CPU: {result.peak_cpu_percent:.1f}%")
        
        # Threshold comparison
        print(f"\n🎯 Threshold Comparison:")
        success_rate = result.successful_requests / result.total_requests if result.total_requests > 0 else 0
        success_icon = "✅" if success_rate >= config.success_rate_threshold else "❌"
        print(f"   {success_icon} Success Rate: {success_rate:.2%} (threshold: {config.success_rate_threshold:.2%})")
        
        p95_icon = "✅" if result.p95_response_time_ms <= config.response_time_p95_ms else "❌"
        print(f"   {p95_icon} P95 Response Time: {result.p95_response_time_ms:.1f}ms (threshold: {config.response_time_p95_ms}ms)")
        
        memory_icon = "✅" if result.peak_memory_mb <= config.memory_limit_mb else "❌"
        print(f"   {memory_icon} Peak Memory: {result.peak_memory_mb:.1f}MB (limit: {config.memory_limit_mb}MB)")
        
        cpu_icon = "✅" if result.peak_cpu_percent <= config.cpu_limit_percent else "❌"
        print(f"   {cpu_icon} Peak CPU: {result.peak_cpu_percent:.1f}% (limit: {config.cpu_limit_percent}%)")
        
        # Errors (if any)
        if result.failed_requests > 0:
            print(f"\n⚠️  Errors ({result.failed_requests} failed requests):")
            unique_errors = list(set(result.error_details[:5]))
            for i, error in enumerate(unique_errors):
                print(f"   {i+1}. {error}")


def run_comprehensive_load_tests():
    """Run comprehensive load testing suite."""
    tester = LoadTester()
    
    print("🚀 Starting Comprehensive Load Testing Suite")
    print("=" * 60)
    
    # Define test configurations
    test_configs = [
        LoadTestConfig(
            test_name="Light Load - Anomaly Detection",
            duration_seconds=120,
            concurrent_users=5,
            requests_per_second=10,
            data_size_mb=1,
            success_rate_threshold=0.95,
            response_time_p95_ms=500
        ),
        LoadTestConfig(
            test_name="Medium Load - Anomaly Detection",
            duration_seconds=180,
            concurrent_users=15,
            requests_per_second=25,
            data_size_mb=5,
            success_rate_threshold=0.90,
            response_time_p95_ms=1000
        ),
        LoadTestConfig(
            test_name="Heavy Load - Anomaly Detection",
            duration_seconds=240,
            concurrent_users=30,
            requests_per_second=50,
            data_size_mb=10,
            success_rate_threshold=0.85,
            response_time_p95_ms=2000,
            memory_limit_mb=4096
        ),
        LoadTestConfig(
            test_name="Data Quality - Production Scale",
            duration_seconds=180,
            concurrent_users=20,
            requests_per_second=30,
            data_size_mb=20,
            success_rate_threshold=0.90,
            response_time_p95_ms=1500
        )
    ]
    
    results = []
    
    for config in test_configs:
        if "Anomaly Detection" in config.test_name:
            result = tester.run_anomaly_detection_load_test(config)
        else:
            result = tester.run_data_quality_load_test(config)
            
        results.append(result)
        
        print("\n" + "-" * 60 + "\n")
    
    # Generate summary report
    print("📋 LOAD TESTING SUMMARY")
    print("=" * 60)
    
    passed_tests = sum(1 for r in results if r.success)
    total_tests = len(results)
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests} ✅")
    print(f"Failed: {total_tests - passed_tests} ❌")
    print(f"Success Rate: {(passed_tests/total_tests*100):.1f}%")
    
    # Performance summary
    avg_response_time = sum(r.average_response_time_ms for r in results) / len(results)
    max_memory = max(r.peak_memory_mb for r in results)
    max_cpu = max(r.peak_cpu_percent for r in results)
    total_requests = sum(r.total_requests for r in results)
    
    print(f"\n📊 Performance Summary:")
    print(f"   Total Requests Processed: {total_requests}")
    print(f"   Average Response Time: {avg_response_time:.1f}ms")
    print(f"   Peak Memory Usage: {max_memory:.1f}MB")
    print(f"   Peak CPU Usage: {max_cpu:.1f}%")
    
    # Save results
    results_data = [
        {
            'test_name': r.test_name,
            'success': r.success,
            'duration_seconds': r.duration_seconds,
            'total_requests': r.total_requests,
            'success_rate': r.successful_requests / r.total_requests if r.total_requests > 0 else 0,
            'avg_response_time_ms': r.average_response_time_ms,
            'p95_response_time_ms': r.p95_response_time_ms,
            'requests_per_second': r.requests_per_second,
            'peak_memory_mb': r.peak_memory_mb,
            'peak_cpu_percent': r.peak_cpu_percent
        }
        for r in results
    ]
    
    with open('LOAD_TEST_RESULTS.json', 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\n💾 Results saved to: LOAD_TEST_RESULTS.json")
    
    return results


if __name__ == "__main__":
    try:
        results = run_comprehensive_load_tests()
        
        # Exit code based on test results
        failed_tests = sum(1 for r in results if not r.success)
        exit_code = 0 if failed_tests == 0 else 1
        
        print(f"\n🏁 Load testing completed with exit code {exit_code}")
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\n⚠️ Load testing interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Load testing failed with error: {e}")
        sys.exit(1)