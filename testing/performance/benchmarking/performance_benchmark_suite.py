import asyncio
import time
import statistics
import json
import random
import gc
import psutil
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import concurrent.futures
import threading
import logging
from contextlib import contextmanager
from prometheus_client import Counter, Histogram, Gauge
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkConfig:
    name: str
    description: str
    iterations: int
    warmup_iterations: int
    timeout_seconds: int
    memory_limit_mb: int
    parallel_workers: int = 1
    enable_profiling: bool = False

@dataclass
class BenchmarkResult:
    benchmark_name: str
    success: bool
    execution_time_seconds: float
    memory_usage_mb: float
    cpu_usage_percent: float
    throughput_ops_per_second: Optional[float] = None
    error_message: Optional[str] = None
    timestamp: datetime = None
    metrics: Dict[str, Any] = None

@dataclass
class BenchmarkSuite:
    name: str
    description: str
    benchmarks: List[BenchmarkConfig]
    system_requirements: Dict[str, Any]

class PerformanceBenchmarker:
    def __init__(self):
        self.results: Dict[str, List[BenchmarkResult]] = {}
        self.system_info: Dict[str, Any] = {}
        
        # Metrics
        self.benchmark_counter = Counter('performance_benchmark_executions_total',
                                       'Total benchmark executions',
                                       ['benchmark', 'status'])
        self.benchmark_duration = Histogram('performance_benchmark_duration_seconds',
                                          'Benchmark execution duration',
                                          ['benchmark'])
        self.memory_usage_gauge = Gauge('performance_benchmark_memory_usage_mb',
                                      'Memory usage during benchmark',
                                      ['benchmark'])
        
        # Collect system information
        self._collect_system_info()
        
        logger.info("Performance Benchmarker initialized")

    def _collect_system_info(self):
        """Collect system information for benchmarking context"""
        try:
            self.system_info = {
                'cpu_count': psutil.cpu_count(),
                'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'platform': f"{psutil.os.name}",
                'python_version': f"{psutil.sys.version}",
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to collect system info: {e}")
            self.system_info = {'error': str(e)}

    async def run_benchmark_suite(self, suite: BenchmarkSuite) -> Dict[str, List[BenchmarkResult]]:
        """Run a complete benchmark suite"""
        logger.info(f"Starting benchmark suite: {suite.name}")
        
        suite_results = {}
        
        for benchmark_config in suite.benchmarks:
            logger.info(f"Running benchmark: {benchmark_config.name}")
            
            # Check system requirements
            if not self._check_system_requirements(suite.system_requirements):
                logger.warning(f"System requirements not met for {benchmark_config.name}")
                continue
            
            # Run benchmark
            benchmark_results = await self._run_benchmark(benchmark_config)
            suite_results[benchmark_config.name] = benchmark_results
            
            # Store results
            if benchmark_config.name not in self.results:
                self.results[benchmark_config.name] = []
            self.results[benchmark_config.name].extend(benchmark_results)
            
            # Allow system to recover between benchmarks
            await asyncio.sleep(2)
        
        logger.info(f"Completed benchmark suite: {suite.name}")
        return suite_results

    def _check_system_requirements(self, requirements: Dict[str, Any]) -> bool:
        """Check if system meets benchmark requirements"""
        try:
            if 'min_memory_gb' in requirements:
                if self.system_info.get('memory_total_gb', 0) < requirements['min_memory_gb']:
                    return False
            
            if 'min_cpu_cores' in requirements:
                if self.system_info.get('cpu_count', 0) < requirements['min_cpu_cores']:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking system requirements: {e}")
            return False

    async def _run_benchmark(self, config: BenchmarkConfig) -> List[BenchmarkResult]:
        """Run a single benchmark with multiple iterations"""
        results = []
        
        # Warmup iterations
        logger.info(f"Running {config.warmup_iterations} warmup iterations for {config.name}")
        for i in range(config.warmup_iterations):
            try:
                await self._execute_benchmark_iteration(config, warmup=True)
            except Exception as e:
                logger.warning(f"Warmup iteration {i} failed: {e}")
        
        # Actual benchmark iterations
        logger.info(f"Running {config.iterations} benchmark iterations for {config.name}")
        for i in range(config.iterations):
            try:
                result = await self._execute_benchmark_iteration(config, warmup=False)
                results.append(result)
                
                # Update metrics
                status = "success" if result.success else "error"
                self.benchmark_counter.labels(
                    benchmark=config.name,
                    status=status
                ).inc()
                
                if result.success:
                    self.benchmark_duration.labels(
                        benchmark=config.name
                    ).observe(result.execution_time_seconds)
                    
                    self.memory_usage_gauge.labels(
                        benchmark=config.name
                    ).set(result.memory_usage_mb)
                
            except Exception as e:
                logger.error(f"Benchmark iteration {i} failed: {e}")
                results.append(BenchmarkResult(
                    benchmark_name=config.name,
                    success=False,
                    execution_time_seconds=0,
                    memory_usage_mb=0,
                    cpu_usage_percent=0,
                    error_message=str(e),
                    timestamp=datetime.utcnow()
                ))
        
        return results

    async def _execute_benchmark_iteration(self, config: BenchmarkConfig, 
                                         warmup: bool = False) -> BenchmarkResult:
        """Execute a single benchmark iteration"""
        # Force garbage collection before benchmark
        gc.collect()
        
        # Monitor system resources
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        start_time = time.time()
        cpu_start = time.time()
        
        try:
            # Execute the benchmark based on configuration
            if config.parallel_workers > 1:
                result_data = await self._execute_parallel_benchmark(config)
            else:
                result_data = await self._execute_sequential_benchmark(config)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Measure resource usage
            final_memory = process.memory_info().rss / (1024 * 1024)  # MB
            memory_usage = final_memory - initial_memory
            
            # Estimate CPU usage (simplified)
            cpu_usage = psutil.cpu_percent(interval=0.1)
            
            # Calculate throughput if applicable
            throughput = None
            if result_data and 'operations_count' in result_data:
                throughput = result_data['operations_count'] / execution_time
            
            return BenchmarkResult(
                benchmark_name=config.name,
                success=True,
                execution_time_seconds=execution_time,
                memory_usage_mb=memory_usage,
                cpu_usage_percent=cpu_usage,
                throughput_ops_per_second=throughput,
                timestamp=datetime.utcnow(),
                metrics=result_data
            )
            
        except asyncio.TimeoutError:
            return BenchmarkResult(
                benchmark_name=config.name,
                success=False,
                execution_time_seconds=time.time() - start_time,
                memory_usage_mb=0,
                cpu_usage_percent=0,
                error_message="Benchmark timeout",
                timestamp=datetime.utcnow()
            )
        except Exception as e:
            return BenchmarkResult(
                benchmark_name=config.name,
                success=False,
                execution_time_seconds=time.time() - start_time,
                memory_usage_mb=0,
                cpu_usage_percent=0,
                error_message=str(e),
                timestamp=datetime.utcnow()
            )

    async def _execute_sequential_benchmark(self, config: BenchmarkConfig) -> Dict[str, Any]:
        """Execute benchmark sequentially"""
        operations_count = 0
        
        # Determine benchmark type and execute accordingly
        if "data_processing" in config.name.lower():
            operations_count = await self._benchmark_data_processing()
        elif "api_gateway" in config.name.lower():
            operations_count = await self._benchmark_api_gateway()
        elif "message_broker" in config.name.lower():
            operations_count = await self._benchmark_message_broker()
        elif "sap_connector" in config.name.lower():
            operations_count = await self._benchmark_sap_connector()
        elif "ml_pipeline" in config.name.lower():
            operations_count = await self._benchmark_ml_pipeline()
        elif "streaming" in config.name.lower():
            operations_count = await self._benchmark_streaming()
        elif "analytics" in config.name.lower():
            operations_count = await self._benchmark_analytics()
        else:
            operations_count = await self._benchmark_generic_workload()
        
        return {
            'operations_count': operations_count,
            'benchmark_type': 'sequential'
        }

    async def _execute_parallel_benchmark(self, config: BenchmarkConfig) -> Dict[str, Any]:
        """Execute benchmark with parallel workers"""
        tasks = []
        
        for worker_id in range(config.parallel_workers):
            task = asyncio.create_task(
                self._execute_sequential_benchmark(config)
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_operations = 0
        successful_workers = 0
        
        for result in results:
            if isinstance(result, dict) and 'operations_count' in result:
                total_operations += result['operations_count']
                successful_workers += 1
        
        return {
            'operations_count': total_operations,
            'successful_workers': successful_workers,
            'benchmark_type': 'parallel'
        }

    # Specific benchmark implementations
    
    async def _benchmark_data_processing(self) -> int:
        """Benchmark data processing operations"""
        operations_count = 0
        
        # Simulate data processing workload
        data_size = 100000
        data = np.random.randn(data_size, 10)
        
        # Perform various data operations
        for _ in range(100):
            # Data transformation
            transformed = np.sqrt(np.abs(data))
            
            # Statistical operations
            means = np.mean(transformed, axis=0)
            stds = np.std(transformed, axis=0)
            
            # Aggregations
            sums = np.sum(transformed, axis=1)
            
            operations_count += 3  # 3 operations per iteration
            
            # Brief pause to allow other operations
            if operations_count % 10 == 0:
                await asyncio.sleep(0.001)
        
        return operations_count

    async def _benchmark_api_gateway(self) -> int:
        """Benchmark API gateway operations"""
        operations_count = 0
        
        # Simulate API gateway workload
        for _ in range(1000):
            # Simulate request processing
            request_data = {
                'path': f'/api/v1/resource/{random.randint(1, 1000)}',
                'method': random.choice(['GET', 'POST', 'PUT', 'DELETE']),
                'headers': {
                    'Authorization': f'Bearer token_{random.randint(1, 1000)}',
                    'Content-Type': 'application/json'
                },
                'body': {'data': f'test_data_{random.randint(1, 1000)}'}
            }
            
            # Simulate authentication
            auth_valid = self._simulate_auth_check(request_data['headers'])
            
            # Simulate rate limiting
            rate_limit_ok = self._simulate_rate_limit_check()
            
            # Simulate request transformation
            transformed_request = self._simulate_request_transformation(request_data)
            
            operations_count += 3  # auth + rate limit + transformation
            
            if operations_count % 100 == 0:
                await asyncio.sleep(0.001)
        
        return operations_count

    async def _benchmark_message_broker(self) -> int:
        """Benchmark message broker operations"""
        operations_count = 0
        
        # Simulate message broker workload
        messages = []
        
        # Publish messages
        for i in range(500):
            message = {
                'id': f'msg_{i}',
                'topic': f'topic_{i % 10}',
                'payload': f'data_{random.randint(1, 10000)}',
                'timestamp': time.time()
            }
            messages.append(message)
            operations_count += 1
        
        # Consume messages
        for message in messages:
            # Simulate message processing
            processed = self._simulate_message_processing(message)
            operations_count += 1
            
            if operations_count % 50 == 0:
                await asyncio.sleep(0.001)
        
        return operations_count

    async def _benchmark_sap_connector(self) -> int:
        """Benchmark SAP connector operations"""
        operations_count = 0
        
        # Simulate SAP connector workload
        for _ in range(100):
            # Simulate BAPI call
            bapi_params = {
                'function': 'BAPI_CUSTOMER_GETDETAIL2',
                'parameters': {
                    'CUSTOMERNO': f'{random.randint(100000, 999999)}',
                    'COMPANYCODE': '1000'
                }
            }
            bapi_result = await self._simulate_bapi_call(bapi_params)
            operations_count += 1
            
            # Simulate RFC call
            rfc_params = {
                'function': 'RFC_READ_TABLE',
                'parameters': {
                    'QUERY_TABLE': 'MARA',
                    'ROWCOUNT': 100
                }
            }
            rfc_result = await self._simulate_rfc_call(rfc_params)
            operations_count += 1
            
            # Simulate data transformation
            transformed = self._simulate_sap_data_transformation(bapi_result)
            operations_count += 1
            
            await asyncio.sleep(0.01)  # SAP operations are slower
        
        return operations_count

    async def _benchmark_ml_pipeline(self) -> int:
        """Benchmark ML pipeline operations"""
        operations_count = 0
        
        # Simulate ML pipeline workload
        data_size = 1000
        features = 50
        
        # Generate synthetic data
        X = np.random.randn(data_size, features)
        y = np.random.randint(0, 2, data_size)
        
        # Simulate ML operations
        for _ in range(10):
            # Data preprocessing
            X_scaled = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
            operations_count += 1
            
            # Feature engineering
            X_poly = np.column_stack([X_scaled, X_scaled**2])
            operations_count += 1
            
            # Model training simulation (simplified)
            weights = np.random.randn(X_poly.shape[1])
            predictions = np.dot(X_poly, weights)
            operations_count += 1
            
            # Model evaluation
            accuracy = np.mean((predictions > 0) == y)
            operations_count += 1
            
            await asyncio.sleep(0.01)
        
        return operations_count

    async def _benchmark_streaming(self) -> int:
        """Benchmark streaming operations"""
        operations_count = 0
        
        # Simulate streaming workload
        stream_size = 10000
        
        for i in range(stream_size):
            # Simulate streaming data point
            data_point = {
                'timestamp': time.time(),
                'value': random.uniform(0, 100),
                'metadata': f'stream_data_{i}'
            }
            
            # Simulate stream processing
            processed = self._simulate_stream_processing(data_point)
            operations_count += 1
            
            # Simulate windowing operation
            if i % 100 == 0:
                window_result = self._simulate_windowing_operation(i)
                operations_count += 1
            
            if operations_count % 1000 == 0:
                await asyncio.sleep(0.001)
        
        return operations_count

    async def _benchmark_analytics(self) -> int:
        """Benchmark analytics operations"""
        operations_count = 0
        
        # Generate sample analytics data
        data_size = 50000
        analytics_data = {
            'users': np.random.randint(1, 10000, data_size),
            'sessions': np.random.randint(1, 1000, data_size),
            'events': np.random.randint(1, 100, data_size),
            'revenue': np.random.uniform(0, 1000, data_size)
        }
        
        # Perform analytics operations
        for _ in range(50):
            # Aggregations
            total_users = len(np.unique(analytics_data['users']))
            total_sessions = np.sum(analytics_data['sessions'])
            total_events = np.sum(analytics_data['events'])
            total_revenue = np.sum(analytics_data['revenue'])
            operations_count += 4
            
            # Statistical analysis
            avg_events_per_session = np.mean(analytics_data['events'])
            revenue_percentiles = np.percentile(analytics_data['revenue'], [25, 50, 75, 95])
            operations_count += 2
            
            # Cohort analysis simulation
            cohort_data = self._simulate_cohort_analysis(analytics_data)
            operations_count += 1
            
            await asyncio.sleep(0.001)
        
        return operations_count

    async def _benchmark_generic_workload(self) -> int:
        """Generic benchmark workload"""
        operations_count = 0
        
        # CPU-intensive operations
        for _ in range(10000):
            # Mathematical computations
            result = sum(i**2 for i in range(100))
            operations_count += 1
            
            if operations_count % 1000 == 0:
                await asyncio.sleep(0.001)
        
        return operations_count

    # Helper methods for simulations
    
    def _simulate_auth_check(self, headers: Dict[str, str]) -> bool:
        """Simulate authentication check"""
        # Simulate some processing time
        time.sleep(0.0001)
        return 'Authorization' in headers

    def _simulate_rate_limit_check(self) -> bool:
        """Simulate rate limiting check"""
        # Simulate some processing time
        time.sleep(0.0001)
        return random.random() > 0.01  # 99% pass rate

    def _simulate_request_transformation(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate request transformation"""
        # Simulate some processing
        transformed = request.copy()
        transformed['processed'] = True
        time.sleep(0.0001)
        return transformed

    def _simulate_message_processing(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate message processing"""
        # Simulate processing
        processed = message.copy()
        processed['processed_at'] = time.time()
        time.sleep(0.0001)
        return processed

    async def _simulate_bapi_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate SAP BAPI call"""
        await asyncio.sleep(0.001)  # Simulate network latency
        return {
            'success': True,
            'data': {'customer_id': params['parameters']['CUSTOMERNO']},
            'return_code': 'S'
        }

    async def _simulate_rfc_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate SAP RFC call"""
        await asyncio.sleep(0.001)  # Simulate network latency
        return {
            'success': True,
            'data': [{'field1': 'value1', 'field2': 'value2'}],
            'return_code': '0'
        }

    def _simulate_sap_data_transformation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate SAP data transformation"""
        time.sleep(0.0001)
        return {'transformed': True, 'original': data}

    def _simulate_stream_processing(self, data_point: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate stream processing"""
        processed = data_point.copy()
        processed['processed'] = True
        return processed

    def _simulate_windowing_operation(self, window_id: int) -> Dict[str, Any]:
        """Simulate windowing operation"""
        return {
            'window_id': window_id,
            'aggregated_value': random.uniform(0, 1000),
            'count': 100
        }

    def _simulate_cohort_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate cohort analysis"""
        # Simplified cohort analysis
        return {
            'cohort_size': len(data['users']),
            'retention_rate': random.uniform(0.1, 0.9)
        }

    def generate_benchmark_report(self, output_file: str):
        """Generate comprehensive benchmark report"""
        if not self.results:
            logger.warning("No benchmark results to report")
            return
        
        report_data = self._calculate_benchmark_statistics()
        
        # Generate HTML report
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Performance Benchmark Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .metric {{ margin: 10px 0; }}
        .metric-value {{ font-weight: bold; color: #2c3e50; }}
        .excellent {{ color: #27ae60; }}
        .good {{ color: #3498db; }}
        .warning {{ color: #f39c12; }}
        .poor {{ color: #e74c3c; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .benchmark-section {{ margin: 30px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Performance Benchmark Report</h1>
        <p>Generated: {datetime.utcnow().isoformat()}</p>
        <h3>System Information</h3>
        <div class="metric">CPU Cores: <span class="metric-value">{self.system_info.get('cpu_count', 'Unknown')}</span></div>
        <div class="metric">Total Memory: <span class="metric-value">{self.system_info.get('memory_total_gb', 0):.1f} GB</span></div>
        <div class="metric">Platform: <span class="metric-value">{self.system_info.get('platform', 'Unknown')}</span></div>
    </div>
    
    <div class="benchmark-section">
        <h2>Benchmark Summary</h2>
        <table>
            <tr>
                <th>Benchmark</th>
                <th>Iterations</th>
                <th>Success Rate</th>
                <th>Avg Execution Time (s)</th>
                <th>Avg Memory Usage (MB)</th>
                <th>Avg Throughput (ops/s)</th>
                <th>Performance Grade</th>
            </tr>
            {self._generate_benchmark_summary_table(report_data)}
        </table>
    </div>
    
    <div class="benchmark-section">
        <h2>Performance Analysis</h2>
        {self._generate_performance_analysis(report_data)}
    </div>
    
    <div class="benchmark-section">
        <h2>Recommendations</h2>
        <ul>
            {self._generate_performance_recommendations(report_data)}
        </ul>
    </div>
</body>
</html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Benchmark report generated: {output_file}")

    def _calculate_benchmark_statistics(self) -> Dict[str, Any]:
        """Calculate statistical summary of benchmark results"""
        stats = {}
        
        for benchmark_name, results in self.results.items():
            successful_results = [r for r in results if r.success]
            
            if not successful_results:
                stats[benchmark_name] = {
                    'success_rate': 0,
                    'avg_execution_time': 0,
                    'avg_memory_usage': 0,
                    'avg_throughput': 0,
                    'p95_execution_time': 0,
                    'performance_grade': 'F'
                }
                continue
            
            execution_times = [r.execution_time_seconds for r in successful_results]
            memory_usages = [r.memory_usage_mb for r in successful_results]
            throughputs = [r.throughput_ops_per_second for r in successful_results 
                          if r.throughput_ops_per_second is not None]
            
            avg_execution_time = statistics.mean(execution_times)
            avg_memory_usage = statistics.mean(memory_usages)
            avg_throughput = statistics.mean(throughputs) if throughputs else 0
            
            # Calculate p95 execution time
            execution_times.sort()
            p95_index = int(len(execution_times) * 0.95)
            p95_execution_time = execution_times[p95_index] if execution_times else 0
            
            # Calculate performance grade
            performance_grade = self._calculate_performance_grade(
                avg_execution_time, avg_throughput, benchmark_name
            )
            
            stats[benchmark_name] = {
                'total_iterations': len(results),
                'successful_iterations': len(successful_results),
                'success_rate': len(successful_results) / len(results) * 100,
                'avg_execution_time': avg_execution_time,
                'avg_memory_usage': avg_memory_usage,
                'avg_throughput': avg_throughput,
                'p95_execution_time': p95_execution_time,
                'performance_grade': performance_grade
            }
        
        return stats

    def _calculate_performance_grade(self, execution_time: float, 
                                   throughput: float, benchmark_name: str) -> str:
        """Calculate performance grade based on execution time and throughput"""
        # Define performance thresholds based on benchmark type
        thresholds = {
            'data_processing': {'excellent': 1.0, 'good': 2.0, 'warning': 5.0},
            'api_gateway': {'excellent': 0.1, 'good': 0.5, 'warning': 2.0},
            'message_broker': {'excellent': 0.5, 'good': 1.0, 'warning': 3.0},
            'sap_connector': {'excellent': 2.0, 'good': 5.0, 'warning': 10.0},
            'ml_pipeline': {'excellent': 5.0, 'good': 10.0, 'warning': 30.0},
            'streaming': {'excellent': 1.0, 'good': 3.0, 'warning': 10.0},
            'analytics': {'excellent': 2.0, 'good': 5.0, 'warning': 15.0}
        }
        
        # Determine benchmark category
        category = 'data_processing'  # default
        for cat in thresholds.keys():
            if cat in benchmark_name.lower():
                category = cat
                break
        
        threshold = thresholds[category]
        
        if execution_time <= threshold['excellent']:
            return 'A'
        elif execution_time <= threshold['good']:
            return 'B'
        elif execution_time <= threshold['warning']:
            return 'C'
        else:
            return 'D'

    def _generate_benchmark_summary_table(self, stats: Dict[str, Any]) -> str:
        """Generate HTML table rows for benchmark summary"""
        rows = []
        
        for benchmark_name, data in stats.items():
            grade = data['performance_grade']
            grade_class = {
                'A': 'excellent',
                'B': 'good', 
                'C': 'warning',
                'D': 'poor',
                'F': 'poor'
            }.get(grade, 'poor')
            
            success_rate = data['success_rate']
            success_class = 'excellent' if success_rate > 95 else 'good' if success_rate > 80 else 'warning' if success_rate > 60 else 'poor'
            
            rows.append(f"""
                <tr>
                    <td>{benchmark_name}</td>
                    <td>{data['total_iterations']}</td>
                    <td class="{success_class}">{success_rate:.1f}%</td>
                    <td>{data['avg_execution_time']:.3f}</td>
                    <td>{data['avg_memory_usage']:.1f}</td>
                    <td>{data['avg_throughput']:.1f}</td>
                    <td class="{grade_class}">{grade}</td>
                </tr>
            """)
        
        return ''.join(rows)

    def _generate_performance_analysis(self, stats: Dict[str, Any]) -> str:
        """Generate performance analysis section"""
        analysis = []
        
        # Overall performance summary
        total_benchmarks = len(stats)
        excellent_count = sum(1 for s in stats.values() if s['performance_grade'] == 'A')
        good_count = sum(1 for s in stats.values() if s['performance_grade'] == 'B')
        
        analysis.append(f"""
        <h3>Overall Performance</h3>
        <div class="metric">Total Benchmarks: <span class="metric-value">{total_benchmarks}</span></div>
        <div class="metric">Excellent Performance (A): <span class="metric-value excellent">{excellent_count}</span></div>
        <div class="metric">Good Performance (B): <span class="metric-value good">{good_count}</span></div>
        <div class="metric">Overall Score: <span class="metric-value">{((excellent_count * 4 + good_count * 3) / (total_benchmarks * 4) * 100):.1f}%</span></div>
        """)
        
        # Best and worst performers
        if stats:
            best_performer = min(stats.items(), key=lambda x: x[1]['avg_execution_time'])
            worst_performer = max(stats.items(), key=lambda x: x[1]['avg_execution_time'])
            
            analysis.append(f"""
            <h3>Performance Highlights</h3>
            <div class="metric">Best Performer: <span class="metric-value excellent">{best_performer[0]}</span> ({best_performer[1]['avg_execution_time']:.3f}s)</div>
            <div class="metric">Needs Attention: <span class="metric-value warning">{worst_performer[0]}</span> ({worst_performer[1]['avg_execution_time']:.3f}s)</div>
            """)
        
        return ''.join(analysis)

    def _generate_performance_recommendations(self, stats: Dict[str, Any]) -> str:
        """Generate performance recommendations"""
        recommendations = []
        
        for benchmark_name, data in stats.items():
            if data['performance_grade'] in ['D', 'F']:
                recommendations.append(f"<li class='poor'><strong>{benchmark_name}</strong>: Poor performance detected. Consider optimization, scaling, or architectural improvements.</li>")
            elif data['performance_grade'] == 'C':
                recommendations.append(f"<li class='warning'><strong>{benchmark_name}</strong>: Performance needs improvement. Review bottlenecks and optimize critical paths.</li>")
            elif data['success_rate'] < 90:
                recommendations.append(f"<li class='warning'><strong>{benchmark_name}</strong>: Low success rate ({data['success_rate']:.1f}%). Investigate error causes and improve reliability.</li>")
            
            if data['avg_memory_usage'] > 1000:  # > 1GB
                recommendations.append(f"<li class='warning'><strong>{benchmark_name}</strong>: High memory usage ({data['avg_memory_usage']:.1f}MB). Consider memory optimization.</li>")
        
        # General recommendations
        avg_grade = statistics.mean([
            {'A': 4, 'B': 3, 'C': 2, 'D': 1, 'F': 0}[s['performance_grade']] 
            for s in stats.values()
        ])
        
        if avg_grade >= 3.5:
            recommendations.append("<li class='excellent'>Excellent overall performance! System is well-optimized for current workloads.</li>")
        elif avg_grade >= 2.5:
            recommendations.append("<li class='good'>Good performance overall. Consider targeted optimizations for specific components.</li>")
        else:
            recommendations.append("<li class='poor'>Performance needs significant improvement. Consider comprehensive optimization strategy.</li>")
        
        if not recommendations:
            recommendations.append("<li>No specific recommendations at this time. Performance appears acceptable.</li>")
        
        return ''.join(recommendations)

    def export_benchmark_data(self, output_file: str):
        """Export benchmark data in JSON format"""
        export_data = {
            'system_info': self.system_info,
            'benchmark_results': {
                name: [asdict(result) for result in results]
                for name, results in self.results.items()
            },
            'statistics': self._calculate_benchmark_statistics(),
            'export_timestamp': datetime.utcnow().isoformat()
        }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Benchmark data exported to: {output_file}")

# Pre-defined benchmark suites

def create_platform_benchmark_suite() -> BenchmarkSuite:
    """Create comprehensive platform benchmark suite"""
    return BenchmarkSuite(
        name="Enterprise Platform Benchmark Suite",
        description="Comprehensive performance benchmarks for the enterprise platform",
        benchmarks=[
            BenchmarkConfig(
                name="api_gateway_performance",
                description="API Gateway request processing performance",
                iterations=10,
                warmup_iterations=3,
                timeout_seconds=60,
                memory_limit_mb=2048,
                parallel_workers=1
            ),
            BenchmarkConfig(
                name="message_broker_throughput",
                description="Message broker publish/consume throughput",
                iterations=10,
                warmup_iterations=3,
                timeout_seconds=120,
                memory_limit_mb=4096,
                parallel_workers=1
            ),
            BenchmarkConfig(
                name="sap_connector_performance", 
                description="SAP connector integration performance",
                iterations=5,
                warmup_iterations=2,
                timeout_seconds=180,
                memory_limit_mb=1024,
                parallel_workers=1
            ),
            BenchmarkConfig(
                name="ml_pipeline_execution",
                description="ML pipeline training and inference performance",
                iterations=5,
                warmup_iterations=1,
                timeout_seconds=300,
                memory_limit_mb=8192,
                parallel_workers=1
            ),
            BenchmarkConfig(
                name="streaming_data_processing",
                description="Real-time streaming data processing performance",
                iterations=10,
                warmup_iterations=3,
                timeout_seconds=120,
                memory_limit_mb=2048,
                parallel_workers=1
            ),
            BenchmarkConfig(
                name="analytics_computation",
                description="Analytics and reporting computation performance",
                iterations=8,
                warmup_iterations=2,
                timeout_seconds=180,
                memory_limit_mb=4096,
                parallel_workers=1
            ),
            BenchmarkConfig(
                name="data_processing_batch",
                description="Batch data processing performance",
                iterations=8,
                warmup_iterations=2,
                timeout_seconds=240,
                memory_limit_mb=6144,
                parallel_workers=1
            )
        ],
        system_requirements={
            'min_memory_gb': 8,
            'min_cpu_cores': 4
        }
    )

def create_scalability_benchmark_suite() -> BenchmarkSuite:
    """Create scalability-focused benchmark suite"""
    return BenchmarkSuite(
        name="Scalability Benchmark Suite",
        description="Benchmark suite focused on parallel processing and scalability",
        benchmarks=[
            BenchmarkConfig(
                name="parallel_api_gateway",
                description="Parallel API Gateway processing",
                iterations=5,
                warmup_iterations=2,
                timeout_seconds=120,
                memory_limit_mb=4096,
                parallel_workers=4
            ),
            BenchmarkConfig(
                name="parallel_message_processing",
                description="Parallel message processing",
                iterations=5,
                warmup_iterations=2,
                timeout_seconds=180,
                memory_limit_mb=6144,
                parallel_workers=8
            ),
            BenchmarkConfig(
                name="parallel_data_processing",
                description="Parallel data processing workload",
                iterations=5,
                warmup_iterations=1,
                timeout_seconds=240,
                memory_limit_mb=8192,
                parallel_workers=6
            )
        ],
        system_requirements={
            'min_memory_gb': 16,
            'min_cpu_cores': 8
        }
    )

# Example usage
async def main():
    benchmarker = PerformanceBenchmarker()
    
    # Run platform benchmark suite
    platform_suite = create_platform_benchmark_suite()
    results = await benchmarker.run_benchmark_suite(platform_suite)
    
    # Generate reports
    benchmarker.generate_benchmark_report("performance_benchmark_report.html")
    benchmarker.export_benchmark_data("performance_benchmark_data.json")
    
    print("Benchmark suite completed!")
    print(f"Results for {len(results)} benchmarks:")
    
    for benchmark_name, benchmark_results in results.items():
        successful = len([r for r in benchmark_results if r.success])
        total = len(benchmark_results)
        avg_time = statistics.mean([r.execution_time_seconds for r in benchmark_results if r.success])
        
        print(f"  {benchmark_name}: {successful}/{total} successful, avg time: {avg_time:.3f}s")

if __name__ == "__main__":
    asyncio.run(main())