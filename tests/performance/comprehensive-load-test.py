#!/usr/bin/env python3

"""
Comprehensive Load Testing Framework for MLOps Platform
This script performs various types of performance testing including load, stress, and spike testing
"""

import asyncio
import aiohttp
import time
import json
import statistics
import argparse
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import yaml
import random
import string

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Individual test result"""
    timestamp: float
    endpoint: str
    method: str
    status_code: int
    response_time: float
    success: bool
    error_message: Optional[str] = None
    response_size: int = 0

@dataclass
class LoadTestConfig:
    """Load test configuration"""
    base_url: str
    concurrent_users: int
    duration_seconds: int
    ramp_up_seconds: int
    test_scenarios: List[Dict[str, Any]]
    think_time_min: float = 1.0
    think_time_max: float = 3.0

@dataclass
class PerformanceMetrics:
    """Performance metrics summary"""
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time: float
    median_response_time: float
    p95_response_time: float
    p99_response_time: float
    min_response_time: float
    max_response_time: float
    requests_per_second: float
    error_rate: float
    throughput_mb_per_sec: float

@dataclass
class LoadTestReport:
    """Complete load test report"""
    test_name: str
    start_time: str
    end_time: str
    duration: float
    config: LoadTestConfig
    overall_metrics: PerformanceMetrics
    endpoint_metrics: Dict[str, PerformanceMetrics]
    error_summary: Dict[str, int]
    recommendations: List[str]

class LoadTestScenarios:
    """Predefined test scenarios for MLOps platform"""
    
    @staticmethod
    def api_health_check():
        return {
            "name": "health_check",
            "method": "GET",
            "endpoint": "/health",
            "weight": 10,
            "headers": {},
            "data": None
        }
    
    @staticmethod
    def user_authentication():
        return {
            "name": "user_login",
            "method": "POST",
            "endpoint": "/api/v1/auth/login",
            "weight": 5,
            "headers": {"Content-Type": "application/json"},
            "data": {"username": "testuser", "password": "testpass123"}
        }
    
    @staticmethod
    def model_prediction():
        return {
            "name": "model_prediction",
            "method": "POST",
            "endpoint": "/api/v1/models/predict",
            "weight": 30,
            "headers": {
                "Content-Type": "application/json",
                "Authorization": "Bearer test-token"
            },
            "data": {
                "model_id": "test-model-v1",
                "features": [[1.2, 3.4, 5.6, 7.8, 9.0]]
            }
        }
    
    @staticmethod
    def batch_prediction():
        return {
            "name": "batch_prediction",
            "method": "POST",
            "endpoint": "/api/v1/models/batch-predict",
            "weight": 15,
            "headers": {
                "Content-Type": "application/json",
                "Authorization": "Bearer test-token"
            },
            "data": {
                "model_id": "test-model-v1",
                "features": [[random.uniform(0, 10) for _ in range(5)] for _ in range(100)]
            }
        }
    
    @staticmethod
    def model_metrics():
        return {
            "name": "model_metrics",
            "method": "GET",
            "endpoint": "/api/v1/models/test-model-v1/metrics",
            "weight": 8,
            "headers": {"Authorization": "Bearer test-token"},
            "data": None
        }
    
    @staticmethod
    def data_upload():
        return {
            "name": "data_upload",
            "method": "POST",
            "endpoint": "/api/v1/data/upload",
            "weight": 5,
            "headers": {
                "Content-Type": "application/json",
                "Authorization": "Bearer test-token"
            },
            "data": {
                "dataset_name": f"test-dataset-{random.randint(1000, 9999)}",
                "data": [[random.uniform(0, 100) for _ in range(10)] for _ in range(50)]
            }
        }
    
    @staticmethod
    def analytics_query():
        return {
            "name": "analytics_query",
            "method": "POST",
            "endpoint": "/api/v1/analytics/query",
            "weight": 12,
            "headers": {
                "Content-Type": "application/json",
                "Authorization": "Bearer test-token"
            },
            "data": {
                "query_type": "model_performance",
                "filters": {
                    "model_id": "test-model-v1",
                    "time_range": "24h"
                }
            }
        }
    
    @staticmethod
    def user_profile():
        return {
            "name": "user_profile",
            "method": "GET",
            "endpoint": "/api/v1/users/me",
            "weight": 7,
            "headers": {"Authorization": "Bearer test-token"},
            "data": None
        }

class VirtualUser:
    """Represents a virtual user for load testing"""
    
    def __init__(self, user_id: int, session: aiohttp.ClientSession, config: LoadTestConfig):
        self.user_id = user_id
        self.session = session
        self.config = config
        self.results: List[TestResult] = []
        
    async def run_scenario(self, scenario: Dict[str, Any]) -> TestResult:
        """Run a single test scenario"""
        start_time = time.time()
        
        try:
            url = f"{self.config.base_url}{scenario['endpoint']}"
            method = scenario['method'].upper()
            headers = scenario.get('headers', {})
            data = scenario.get('data')
            
            # Add some randomization to data if it's a dict
            if isinstance(data, dict) and 'features' in data:
                if isinstance(data['features'][0], list):
                    # Add noise to feature values
                    data['features'] = [
                        [val + random.uniform(-0.1, 0.1) for val in row] 
                        for row in data['features']
                    ]
            
            async with self.session.request(
                method=method,
                url=url,
                headers=headers,
                json=data if data else None,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                response_text = await response.text()
                response_time = time.time() - start_time
                
                return TestResult(
                    timestamp=start_time,
                    endpoint=scenario['endpoint'],
                    method=method,
                    status_code=response.status,
                    response_time=response_time,
                    success=200 <= response.status < 400,
                    response_size=len(response_text.encode('utf-8'))
                )
                
        except asyncio.TimeoutError:
            return TestResult(
                timestamp=start_time,
                endpoint=scenario['endpoint'],
                method=scenario['method'],
                status_code=0,
                response_time=time.time() - start_time,
                success=False,
                error_message="Timeout"
            )
        except Exception as e:
            return TestResult(
                timestamp=start_time,
                endpoint=scenario['endpoint'],
                method=scenario['method'],
                status_code=0,
                response_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
    
    async def simulate_user_behavior(self) -> List[TestResult]:
        """Simulate realistic user behavior"""
        results = []
        end_time = time.time() + self.config.duration_seconds
        
        while time.time() < end_time:
            # Select scenario based on weights
            total_weight = sum(scenario['weight'] for scenario in self.config.test_scenarios)
            random_weight = random.uniform(0, total_weight)
            
            cumulative_weight = 0
            selected_scenario = None
            for scenario in self.config.test_scenarios:
                cumulative_weight += scenario['weight']
                if random_weight <= cumulative_weight:
                    selected_scenario = scenario
                    break
            
            if selected_scenario:
                result = await self.run_scenario(selected_scenario)
                results.append(result)
                
                # Think time between requests
                think_time = random.uniform(
                    self.config.think_time_min,
                    self.config.think_time_max
                )
                await asyncio.sleep(think_time)
        
        return results

class LoadTester:
    """Main load testing class"""
    
    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.results: List[TestResult] = []
        
    async def run_load_test(self) -> List[TestResult]:
        """Run the load test with specified configuration"""
        logger.info(f"Starting load test with {self.config.concurrent_users} concurrent users")
        logger.info(f"Test duration: {self.config.duration_seconds} seconds")
        logger.info(f"Ramp-up period: {self.config.ramp_up_seconds} seconds")
        
        # Create HTTP session with connection pooling
        connector = aiohttp.TCPConnector(
            limit=self.config.concurrent_users * 2,
            limit_per_host=self.config.concurrent_users,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=60)
        ) as session:
            # Calculate ramp-up delay between users
            ramp_up_delay = self.config.ramp_up_seconds / self.config.concurrent_users
            
            # Start virtual users with staggered ramp-up
            tasks = []
            for user_id in range(self.config.concurrent_users):
                # Wait for ramp-up delay
                await asyncio.sleep(ramp_up_delay)
                
                # Create and start virtual user
                user = VirtualUser(user_id, session, self.config)
                task = asyncio.create_task(user.simulate_user_behavior())
                tasks.append(task)
                
                logger.info(f"Started user {user_id + 1}/{self.config.concurrent_users}")
            
            # Wait for all users to complete
            logger.info("All users started, waiting for test completion...")
            all_results = await asyncio.gather(*tasks)
            
            # Flatten results
            for user_results in all_results:
                self.results.extend(user_results)
        
        logger.info(f"Load test completed. Total requests: {len(self.results)}")
        return self.results
    
    def calculate_metrics(self, results: List[TestResult]) -> PerformanceMetrics:
        """Calculate performance metrics from test results"""
        if not results:
            return PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        response_times = [r.response_time for r in successful_results]
        response_sizes = [r.response_size for r in successful_results]
        
        if response_times:
            avg_response_time = statistics.mean(response_times)
            median_response_time = statistics.median(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            
            # Calculate percentiles
            sorted_times = sorted(response_times)
            p95_index = int(len(sorted_times) * 0.95)
            p99_index = int(len(sorted_times) * 0.99)
            p95_response_time = sorted_times[p95_index] if p95_index < len(sorted_times) else sorted_times[-1]
            p99_response_time = sorted_times[p99_index] if p99_index < len(sorted_times) else sorted_times[-1]
        else:
            avg_response_time = median_response_time = min_response_time = max_response_time = 0
            p95_response_time = p99_response_time = 0
        
        # Calculate throughput
        if results:
            test_duration = max(r.timestamp for r in results) - min(r.timestamp for r in results)
            requests_per_second = len(successful_results) / test_duration if test_duration > 0 else 0
            total_data_mb = sum(response_sizes) / (1024 * 1024)
            throughput_mb_per_sec = total_data_mb / test_duration if test_duration > 0 else 0
        else:
            requests_per_second = throughput_mb_per_sec = 0
        
        error_rate = len(failed_results) / len(results) if results else 0
        
        return PerformanceMetrics(
            total_requests=len(results),
            successful_requests=len(successful_results),
            failed_requests=len(failed_results),
            avg_response_time=avg_response_time,
            median_response_time=median_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            requests_per_second=requests_per_second,
            error_rate=error_rate,
            throughput_mb_per_sec=throughput_mb_per_sec
        )
    
    def generate_report(self, test_name: str) -> LoadTestReport:
        """Generate comprehensive load test report"""
        start_time = min(r.timestamp for r in self.results) if self.results else time.time()
        end_time = max(r.timestamp for r in self.results) if self.results else time.time()
        duration = end_time - start_time
        
        # Calculate overall metrics
        overall_metrics = self.calculate_metrics(self.results)
        
        # Calculate per-endpoint metrics
        endpoint_metrics = {}
        endpoints = set(r.endpoint for r in self.results)
        for endpoint in endpoints:
            endpoint_results = [r for r in self.results if r.endpoint == endpoint]
            endpoint_metrics[endpoint] = self.calculate_metrics(endpoint_results)
        
        # Error summary
        error_summary = {}
        for result in self.results:
            if not result.success and result.error_message:
                error_summary[result.error_message] = error_summary.get(result.error_message, 0) + 1
        
        # Generate recommendations
        recommendations = self._generate_recommendations(overall_metrics, endpoint_metrics)
        
        return LoadTestReport(
            test_name=test_name,
            start_time=datetime.fromtimestamp(start_time).isoformat(),
            end_time=datetime.fromtimestamp(end_time).isoformat(),
            duration=duration,
            config=self.config,
            overall_metrics=overall_metrics,
            endpoint_metrics=endpoint_metrics,
            error_summary=error_summary,
            recommendations=recommendations
        )
    
    def _generate_recommendations(self, overall_metrics: PerformanceMetrics, 
                                 endpoint_metrics: Dict[str, PerformanceMetrics]) -> List[str]:
        """Generate performance recommendations based on test results"""
        recommendations = []
        
        # Response time recommendations
        if overall_metrics.avg_response_time > 1.0:
            recommendations.append("High average response time detected. Consider optimizing application performance.")
        
        if overall_metrics.p95_response_time > 2.0:
            recommendations.append("95th percentile response time is high. Investigate slow requests and optimize bottlenecks.")
        
        # Error rate recommendations
        if overall_metrics.error_rate > 0.05:  # 5% error rate
            recommendations.append("High error rate detected. Investigate error causes and improve error handling.")
        
        # Throughput recommendations
        if overall_metrics.requests_per_second < 100:
            recommendations.append("Low throughput detected. Consider scaling horizontally or optimizing performance.")
        
        # Endpoint-specific recommendations
        for endpoint, metrics in endpoint_metrics.items():
            if metrics.avg_response_time > 2.0:
                recommendations.append(f"Endpoint {endpoint} has high response times. Consider caching or optimization.")
            
            if metrics.error_rate > 0.1:  # 10% error rate for specific endpoint
                recommendations.append(f"Endpoint {endpoint} has high error rate. Review implementation and error handling.")
        
        # General recommendations
        if not recommendations:
            recommendations.append("Performance looks good! Consider running stress tests to find breaking points.")
        
        return recommendations

class LoadTestReportGenerator:
    """Generate various formats of load test reports"""
    
    @staticmethod
    def generate_json_report(report: LoadTestReport, output_file: str):
        """Generate JSON report"""
        with open(output_file, 'w') as f:
            json.dump(asdict(report), f, indent=2, default=str)
    
    @staticmethod
    def generate_html_report(report: LoadTestReport, output_file: str):
        """Generate HTML report with charts"""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Load Test Report - {report.test_name}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background: #f4f4f4; padding: 20px; border-radius: 5px; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric {{ background: #f9f9f9; padding: 15px; border-radius: 5px; text-align: center; }}
        .chart-container {{ margin: 20px 0; height: 400px; }}
        .recommendations {{ background: #e8f5e8; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .endpoint {{ margin: 10px 0; padding: 10px; background: #f0f0f0; border-radius: 3px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Load Test Report: {report.test_name}</h1>
        <p><strong>Start Time:</strong> {report.start_time}</p>
        <p><strong>Duration:</strong> {report.duration:.2f} seconds</p>
        <p><strong>Concurrent Users:</strong> {report.config.concurrent_users}</p>
        <p><strong>Base URL:</strong> {report.config.base_url}</p>
    </div>
    
    <h2>Overall Performance Metrics</h2>
    <div class="metrics">
        <div class="metric">
            <h3>{report.overall_metrics.total_requests}</h3>
            <p>Total Requests</p>
        </div>
        <div class="metric">
            <h3>{report.overall_metrics.successful_requests}</h3>
            <p>Successful</p>
        </div>
        <div class="metric">
            <h3>{report.overall_metrics.failed_requests}</h3>
            <p>Failed</p>
        </div>
        <div class="metric">
            <h3>{report.overall_metrics.avg_response_time:.3f}s</h3>
            <p>Avg Response Time</p>
        </div>
        <div class="metric">
            <h3>{report.overall_metrics.p95_response_time:.3f}s</h3>
            <p>95th Percentile</p>
        </div>
        <div class="metric">
            <h3>{report.overall_metrics.requests_per_second:.1f}</h3>
            <p>Requests/Second</p>
        </div>
        <div class="metric">
            <h3>{report.overall_metrics.error_rate:.2%}</h3>
            <p>Error Rate</p>
        </div>
        <div class="metric">
            <h3>{report.overall_metrics.throughput_mb_per_sec:.2f} MB/s</h3>
            <p>Throughput</p>
        </div>
    </div>
    
    <h2>Endpoint Performance</h2>
"""
        
        for endpoint, metrics in report.endpoint_metrics.items():
            html_content += f"""
    <div class="endpoint">
        <h3>{endpoint}</h3>
        <p><strong>Requests:</strong> {metrics.total_requests} | 
           <strong>Success Rate:</strong> {(1-metrics.error_rate):.2%} | 
           <strong>Avg Response:</strong> {metrics.avg_response_time:.3f}s |
           <strong>95th Percentile:</strong> {metrics.p95_response_time:.3f}s</p>
    </div>
"""
        
        if report.recommendations:
            html_content += """
    <div class="recommendations">
        <h2>Recommendations</h2>
        <ul>
"""
            for rec in report.recommendations:
                html_content += f"<li>{rec}</li>"
            
            html_content += """
        </ul>
    </div>
"""
        
        html_content += """
</body>
</html>
"""
        
        with open(output_file, 'w') as f:
            f.write(html_content)

async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='MLOps Platform Load Tester')
    parser.add_argument('--base-url', required=True, help='Base URL of the application')
    parser.add_argument('--users', type=int, default=10, help='Number of concurrent users')
    parser.add_argument('--duration', type=int, default=60, help='Test duration in seconds')
    parser.add_argument('--ramp-up', type=int, default=10, help='Ramp-up time in seconds')
    parser.add_argument('--output-dir', default='./load-test-reports', help='Output directory for reports')
    parser.add_argument('--test-name', default='load_test', help='Name of the test')
    parser.add_argument('--scenario', choices=['light', 'normal', 'heavy'], default='normal', help='Test scenario intensity')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Configure test scenarios based on intensity
    scenarios = [
        LoadTestScenarios.api_health_check(),
        LoadTestScenarios.user_authentication(),
        LoadTestScenarios.model_prediction(),
        LoadTestScenarios.model_metrics(),
        LoadTestScenarios.user_profile()
    ]
    
    if args.scenario in ['normal', 'heavy']:
        scenarios.extend([
            LoadTestScenarios.batch_prediction(),
            LoadTestScenarios.analytics_query()
        ])
    
    if args.scenario == 'heavy':
        scenarios.append(LoadTestScenarios.data_upload())
    
    # Configure think time based on scenario
    think_time_config = {
        'light': (2.0, 5.0),
        'normal': (1.0, 3.0),
        'heavy': (0.5, 2.0)
    }
    think_min, think_max = think_time_config[args.scenario]
    
    # Create test configuration
    config = LoadTestConfig(
        base_url=args.base_url.rstrip('/'),
        concurrent_users=args.users,
        duration_seconds=args.duration,
        ramp_up_seconds=args.ramp_up,
        test_scenarios=scenarios,
        think_time_min=think_min,
        think_time_max=think_max
    )
    
    # Run load test
    tester = LoadTester(config)
    results = await tester.run_load_test()
    
    # Generate report
    report = tester.generate_report(args.test_name)
    
    # Save reports
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    json_file = output_dir / f'{args.test_name}-{timestamp}.json'
    LoadTestReportGenerator.generate_json_report(report, str(json_file))
    logger.info(f"JSON report saved: {json_file}")
    
    html_file = output_dir / f'{args.test_name}-{timestamp}.html'
    LoadTestReportGenerator.generate_html_report(report, str(html_file))
    logger.info(f"HTML report saved: {html_file}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"LOAD TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Test Name: {report.test_name}")
    print(f"Duration: {report.duration:.2f} seconds")
    print(f"Total Requests: {report.overall_metrics.total_requests}")
    print(f"Successful: {report.overall_metrics.successful_requests}")
    print(f"Failed: {report.overall_metrics.failed_requests}")
    print(f"Success Rate: {(1-report.overall_metrics.error_rate):.2%}")
    print(f"Average Response Time: {report.overall_metrics.avg_response_time:.3f}s")
    print(f"95th Percentile: {report.overall_metrics.p95_response_time:.3f}s")
    print(f"Requests/Second: {report.overall_metrics.requests_per_second:.1f}")
    print(f"Throughput: {report.overall_metrics.throughput_mb_per_sec:.2f} MB/s")
    print(f"{'='*60}")
    
    # Print recommendations
    if report.recommendations:
        print("RECOMMENDATIONS:")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"{i}. {rec}")
        print(f"{'='*60}")
    
    # Exit with appropriate code
    if report.overall_metrics.error_rate > 0.1:  # 10% error rate
        logger.error("High error rate detected!")
        return 1
    elif report.overall_metrics.avg_response_time > 2.0:
        logger.warning("High response times detected!")
        return 2
    else:
        logger.info("Load test completed successfully!")
        return 0

if __name__ == '__main__':
    exit_code = asyncio.run(main())
    exit(exit_code)