import asyncio
import aiohttp
import time
import statistics
import json
import random
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import concurrent.futures
import psutil
import logging
from prometheus_client import Counter, Histogram, Gauge, Summary

logger = logging.getLogger(__name__)

@dataclass
class LoadTestConfig:
    target_url: str
    concurrent_users: int
    requests_per_user: int
    ramp_up_duration: int
    test_duration: int
    think_time_min: float = 0.1
    think_time_max: float = 2.0
    timeout_seconds: int = 30
    
@dataclass
class TestScenario:
    name: str
    weight: float
    endpoint: str
    method: str
    headers: Dict[str, str]
    payload: Optional[Dict[str, Any]] = None
    expected_status: int = 200

@dataclass
class TestResult:
    scenario_name: str
    response_time: float
    status_code: int
    error_message: Optional[str] = None
    timestamp: datetime = None
    bytes_received: int = 0

@dataclass
class LoadTestSummary:
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time: float
    p50_response_time: float
    p95_response_time: float
    p99_response_time: float
    max_response_time: float
    min_response_time: float
    requests_per_second: float
    error_rate: float
    throughput_mbps: float
    cpu_usage_avg: float
    memory_usage_avg: float

class APIGatewayLoadTester:
    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.results: List[TestResult] = []
        self.system_metrics: List[Dict[str, float]] = []
        
        # Metrics
        self.request_counter = Counter('load_test_requests_total', 
                                     'Total load test requests', 
                                     ['scenario', 'status'])
        self.response_time_histogram = Histogram('load_test_response_time_seconds',
                                               'Response time distribution',
                                               ['scenario'])
        self.concurrent_users_gauge = Gauge('load_test_concurrent_users',
                                          'Current concurrent users')
        self.throughput_gauge = Gauge('load_test_throughput_rps',
                                    'Requests per second')
        
        # Test scenarios
        self.scenarios = self._create_test_scenarios()
        
        logger.info(f"Initialized load tester with {config.concurrent_users} users")

    def _create_test_scenarios(self) -> List[TestScenario]:
        """Create realistic test scenarios for API Gateway"""
        scenarios = [
            # User Management Scenarios
            TestScenario(
                name="get_user_profile",
                weight=0.25,
                endpoint="/api/v1/users/profile",
                method="GET",
                headers={
                    "Authorization": "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.test",
                    "Content-Type": "application/json"
                }
            ),
            TestScenario(
                name="update_user_profile",
                weight=0.15,
                endpoint="/api/v1/users/profile",
                method="PUT",
                headers={
                    "Authorization": "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.test",
                    "Content-Type": "application/json"
                },
                payload={
                    "name": "Load Test User",
                    "email": "loadtest@example.com",
                    "preferences": {
                        "notifications": True,
                        "theme": "dark"
                    }
                }
            ),
            
            # Order Processing Scenarios
            TestScenario(
                name="create_order",
                weight=0.20,
                endpoint="/api/v1/orders",
                method="POST",
                headers={
                    "Authorization": "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.test",
                    "Content-Type": "application/json"
                },
                payload={
                    "customer_id": "CUST-12345",
                    "items": [
                        {
                            "product_id": "PROD-001",
                            "quantity": 2,
                            "price": 29.99
                        },
                        {
                            "product_id": "PROD-002", 
                            "quantity": 1,
                            "price": 49.99
                        }
                    ],
                    "shipping_address": {
                        "street": "123 Main St",
                        "city": "Anytown",
                        "state": "CA",
                        "zip": "12345"
                    }
                },
                expected_status=201
            ),
            TestScenario(
                name="get_order_status",
                weight=0.15,
                endpoint="/api/v1/orders/ORD-12345/status",
                method="GET",
                headers={
                    "Authorization": "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.test",
                    "Content-Type": "application/json"
                }
            ),
            
            # Payment Processing Scenarios
            TestScenario(
                name="process_payment",
                weight=0.10,
                endpoint="/api/v1/payments",
                method="POST",
                headers={
                    "Authorization": "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.test",
                    "Content-Type": "application/json"
                },
                payload={
                    "order_id": "ORD-12345",
                    "amount": 109.97,
                    "currency": "USD",
                    "payment_method": {
                        "type": "credit_card",
                        "card_number": "4111111111111111",
                        "expiry_month": 12,
                        "expiry_year": 2025,
                        "cvv": "123"
                    }
                },
                expected_status=201
            ),
            
            # Legacy SOAP Service Scenarios  
            TestScenario(
                name="legacy_soap_request",
                weight=0.05,
                endpoint="/api/v1/legacy/customer-lookup",
                method="POST",
                headers={
                    "Authorization": "Basic dXNlcjpwYXNzd29yZA==",
                    "Content-Type": "application/json"
                },
                payload={
                    "customer_number": "0000100001",
                    "include_history": True
                }
            ),
            
            # Health Check Scenarios
            TestScenario(
                name="health_check",
                weight=0.10,
                endpoint="/health",
                method="GET",
                headers={}
            )
        ]
        
        return scenarios

    async def run_load_test(self) -> LoadTestSummary:
        """Execute the complete load test"""
        logger.info(f"Starting load test with {self.config.concurrent_users} users")
        
        start_time = time.time()
        
        # Start system monitoring
        monitor_task = asyncio.create_task(self._monitor_system_metrics())
        
        # Create semaphore to limit concurrent connections
        semaphore = asyncio.Semaphore(self.config.concurrent_users)
        
        # Create user tasks with ramp-up
        user_tasks = []
        ramp_up_delay = self.config.ramp_up_duration / self.config.concurrent_users
        
        for user_id in range(self.config.concurrent_users):
            delay = user_id * ramp_up_delay
            task = asyncio.create_task(
                self._simulate_user(user_id, semaphore, delay)
            )
            user_tasks.append(task)
        
        # Wait for all users to complete
        await asyncio.gather(*user_tasks, return_exceptions=True)
        
        # Stop monitoring
        monitor_task.cancel()
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Calculate and return summary
        summary = self._calculate_summary(total_duration)
        
        logger.info(f"Load test completed in {total_duration:.2f} seconds")
        logger.info(f"Total requests: {summary.total_requests}")
        logger.info(f"Success rate: {(1 - summary.error_rate) * 100:.2f}%")
        logger.info(f"Average response time: {summary.avg_response_time:.3f}s")
        logger.info(f"Throughput: {summary.requests_per_second:.2f} RPS")
        
        return summary

    async def _simulate_user(self, user_id: int, semaphore: asyncio.Semaphore, 
                           initial_delay: float):
        """Simulate a single user's behavior"""
        await asyncio.sleep(initial_delay)
        
        async with semaphore:
            self.concurrent_users_gauge.inc()
            
            try:
                timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
                
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    for request_num in range(self.config.requests_per_user):
                        # Select scenario based on weights
                        scenario = self._select_scenario()
                        
                        # Execute request
                        result = await self._execute_request(session, scenario, user_id)
                        self.results.append(result)
                        
                        # Update metrics
                        status_label = "success" if result.status_code < 400 else "error"
                        self.request_counter.labels(
                            scenario=scenario.name,
                            status=status_label
                        ).inc()
                        
                        self.response_time_histogram.labels(
                            scenario=scenario.name
                        ).observe(result.response_time)
                        
                        # Think time between requests
                        think_time = random.uniform(
                            self.config.think_time_min,
                            self.config.think_time_max
                        )
                        await asyncio.sleep(think_time)
                        
            except Exception as e:
                logger.error(f"User {user_id} encountered error: {e}")
                
            finally:
                self.concurrent_users_gauge.dec()

    def _select_scenario(self) -> TestScenario:
        """Select a test scenario based on weights"""
        rand = random.random()
        cumulative_weight = 0.0
        
        for scenario in self.scenarios:
            cumulative_weight += scenario.weight
            if rand <= cumulative_weight:
                return scenario
        
        # Fallback to last scenario
        return self.scenarios[-1]

    async def _execute_request(self, session: aiohttp.ClientSession, 
                             scenario: TestScenario, user_id: int) -> TestResult:
        """Execute a single HTTP request"""
        start_time = time.time()
        url = f"{self.config.target_url}{scenario.endpoint}"
        
        try:
            # Add user-specific headers
            headers = scenario.headers.copy()
            headers['X-Load-Test-User'] = str(user_id)
            headers['X-Load-Test-Scenario'] = scenario.name
            
            # Execute request
            async with session.request(
                method=scenario.method,
                url=url,
                headers=headers,
                json=scenario.payload
            ) as response:
                
                response_time = time.time() - start_time
                content = await response.read()
                
                return TestResult(
                    scenario_name=scenario.name,
                    response_time=response_time,
                    status_code=response.status,
                    timestamp=datetime.utcnow(),
                    bytes_received=len(content),
                    error_message=None if response.status < 400 else await response.text()
                )
                
        except asyncio.TimeoutError:
            return TestResult(
                scenario_name=scenario.name,
                response_time=time.time() - start_time,
                status_code=0,
                timestamp=datetime.utcnow(),
                error_message="Request timeout"
            )
        except Exception as e:
            return TestResult(
                scenario_name=scenario.name,
                response_time=time.time() - start_time,
                status_code=0,
                timestamp=datetime.utcnow(),
                error_message=str(e)
            )

    async def _monitor_system_metrics(self):
        """Monitor system resource usage during test"""
        while True:
            try:
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                metrics = {
                    'timestamp': time.time(),
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_available_gb': memory.available / (1024**3)
                }
                
                self.system_metrics.append(metrics)
                
                await asyncio.sleep(5)  # Sample every 5 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error monitoring system metrics: {e}")
                await asyncio.sleep(5)

    def _calculate_summary(self, total_duration: float) -> LoadTestSummary:
        """Calculate test summary statistics"""
        if not self.results:
            return LoadTestSummary(
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                avg_response_time=0,
                p50_response_time=0,
                p95_response_time=0,
                p99_response_time=0,
                max_response_time=0,
                min_response_time=0,
                requests_per_second=0,
                error_rate=1.0,
                throughput_mbps=0,
                cpu_usage_avg=0,
                memory_usage_avg=0
            )
        
        # Response time statistics
        response_times = [r.response_time for r in self.results]
        response_times.sort()
        
        total_requests = len(self.results)
        successful_requests = len([r for r in self.results if r.status_code < 400])
        failed_requests = total_requests - successful_requests
        
        # Calculate percentiles
        def percentile(data: List[float], p: float) -> float:
            if not data:
                return 0.0
            index = int(len(data) * p / 100)
            return data[min(index, len(data) - 1)]
        
        # System metrics averages
        cpu_avg = statistics.mean([m['cpu_percent'] for m in self.system_metrics]) if self.system_metrics else 0
        memory_avg = statistics.mean([m['memory_percent'] for m in self.system_metrics]) if self.system_metrics else 0
        
        # Throughput calculation
        total_bytes = sum(r.bytes_received for r in self.results)
        throughput_mbps = (total_bytes * 8) / (total_duration * 1024 * 1024)  # Convert to Mbps
        
        return LoadTestSummary(
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            avg_response_time=statistics.mean(response_times),
            p50_response_time=percentile(response_times, 50),
            p95_response_time=percentile(response_times, 95),
            p99_response_time=percentile(response_times, 99),
            max_response_time=max(response_times),
            min_response_time=min(response_times),
            requests_per_second=total_requests / total_duration,
            error_rate=failed_requests / total_requests if total_requests > 0 else 0,
            throughput_mbps=throughput_mbps,
            cpu_usage_avg=cpu_avg,
            memory_usage_avg=memory_avg
        )

    def generate_report(self, summary: LoadTestSummary, output_file: str):
        """Generate detailed HTML report"""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>API Gateway Load Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .metric {{ margin: 10px 0; }}
        .metric-value {{ font-weight: bold; color: #2c3e50; }}
        .error {{ color: #e74c3c; }}
        .success {{ color: #27ae60; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .chart {{ margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>API Gateway Load Test Report</h1>
        <p>Generated: {datetime.utcnow().isoformat()}</p>
        <p>Test Configuration: {self.config.concurrent_users} concurrent users, {self.config.requests_per_user} requests per user</p>
    </div>
    
    <h2>Performance Summary</h2>
    <div class="metric">Total Requests: <span class="metric-value">{summary.total_requests}</span></div>
    <div class="metric">Successful Requests: <span class="metric-value success">{summary.successful_requests}</span></div>
    <div class="metric">Failed Requests: <span class="metric-value error">{summary.failed_requests}</span></div>
    <div class="metric">Success Rate: <span class="metric-value">{(1 - summary.error_rate) * 100:.2f}%</span></div>
    <div class="metric">Average Response Time: <span class="metric-value">{summary.avg_response_time:.3f}s</span></div>
    <div class="metric">95th Percentile Response Time: <span class="metric-value">{summary.p95_response_time:.3f}s</span></div>
    <div class="metric">99th Percentile Response Time: <span class="metric-value">{summary.p99_response_time:.3f}s</span></div>
    <div class="metric">Requests per Second: <span class="metric-value">{summary.requests_per_second:.2f}</span></div>
    <div class="metric">Throughput: <span class="metric-value">{summary.throughput_mbps:.2f} Mbps</span></div>
    
    <h2>System Resource Usage</h2>
    <div class="metric">Average CPU Usage: <span class="metric-value">{summary.cpu_usage_avg:.1f}%</span></div>
    <div class="metric">Average Memory Usage: <span class="metric-value">{summary.memory_usage_avg:.1f}%</span></div>
    
    <h2>Response Time Distribution</h2>
    <table>
        <tr>
            <th>Percentile</th>
            <th>Response Time (ms)</th>
        </tr>
        <tr><td>50th (Median)</td><td>{summary.p50_response_time * 1000:.1f}</td></tr>
        <tr><td>95th</td><td>{summary.p95_response_time * 1000:.1f}</td></tr>
        <tr><td>99th</td><td>{summary.p99_response_time * 1000:.1f}</td></tr>
        <tr><td>Maximum</td><td>{summary.max_response_time * 1000:.1f}</td></tr>
        <tr><td>Minimum</td><td>{summary.min_response_time * 1000:.1f}</td></tr>
    </table>
    
    <h2>Scenario Performance</h2>
    <table>
        <tr>
            <th>Scenario</th>
            <th>Requests</th>
            <th>Success Rate</th>
            <th>Avg Response Time (ms)</th>
        </tr>
        {self._generate_scenario_table()}
    </table>
    
    <h2>Recommendations</h2>
    <ul>
        {self._generate_recommendations(summary)}
    </ul>
</body>
</html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Load test report generated: {output_file}")

    def _generate_scenario_table(self) -> str:
        """Generate HTML table rows for scenario performance"""
        scenario_stats = {}
        
        for result in self.results:
            if result.scenario_name not in scenario_stats:
                scenario_stats[result.scenario_name] = {
                    'total': 0,
                    'success': 0,
                    'response_times': []
                }
            
            stats = scenario_stats[result.scenario_name]
            stats['total'] += 1
            if result.status_code < 400:
                stats['success'] += 1
            stats['response_times'].append(result.response_time)
        
        rows = []
        for scenario_name, stats in scenario_stats.items():
            success_rate = (stats['success'] / stats['total'] * 100) if stats['total'] > 0 else 0
            avg_response_time = statistics.mean(stats['response_times']) * 1000 if stats['response_times'] else 0
            
            rows.append(f"""
                <tr>
                    <td>{scenario_name}</td>
                    <td>{stats['total']}</td>
                    <td>{success_rate:.1f}%</td>
                    <td>{avg_response_time:.1f}</td>
                </tr>
            """)
        
        return ''.join(rows)

    def _generate_recommendations(self, summary: LoadTestSummary) -> str:
        """Generate performance recommendations based on results"""
        recommendations = []
        
        if summary.error_rate > 0.05:  # > 5% error rate
            recommendations.append("<li class='error'>High error rate detected. Consider investigating error causes and improving error handling.</li>")
        
        if summary.p95_response_time > 2.0:  # > 2 seconds
            recommendations.append("<li class='error'>95th percentile response time is high. Consider optimizing slow endpoints or adding caching.</li>")
        
        if summary.cpu_usage_avg > 80:
            recommendations.append("<li class='error'>High CPU usage detected. Consider scaling horizontally or optimizing CPU-intensive operations.</li>")
        
        if summary.memory_usage_avg > 80:
            recommendations.append("<li class='error'>High memory usage detected. Consider optimizing memory usage or increasing available memory.</li>")
        
        if summary.requests_per_second < 100:
            recommendations.append("<li>Low throughput detected. Consider investigating bottlenecks in the request processing pipeline.</li>")
        
        if summary.error_rate < 0.01 and summary.p95_response_time < 1.0:
            recommendations.append("<li class='success'>Excellent performance! System is handling load well with low error rate and good response times.</li>")
        
        if not recommendations:
            recommendations.append("<li>Performance appears normal. Continue monitoring and consider testing with higher load.</li>")
        
        return ''.join(recommendations)

    def export_metrics(self, output_file: str):
        """Export detailed metrics in JSON format"""
        metrics_data = {
            'config': asdict(self.config),
            'results': [asdict(result) for result in self.results],
            'system_metrics': self.system_metrics,
            'scenarios': [asdict(scenario) for scenario in self.scenarios]
        }
        
        with open(output_file, 'w') as f:
            json.dump(metrics_data, f, indent=2, default=str)
        
        logger.info(f"Metrics exported to: {output_file}")

# Load test execution scenarios
class LoadTestRunner:
    def __init__(self):
        self.test_configs = self._create_test_configurations()

    def _create_test_configurations(self) -> List[LoadTestConfig]:
        """Create different load test configurations"""
        return [
            # Light load test
            LoadTestConfig(
                target_url="http://api-gateway:8080",
                concurrent_users=10,
                requests_per_user=50,
                ramp_up_duration=30,
                test_duration=300,  # 5 minutes
                think_time_min=0.5,
                think_time_max=2.0
            ),
            
            # Medium load test
            LoadTestConfig(
                target_url="http://api-gateway:8080",
                concurrent_users=50,
                requests_per_user=100,
                ramp_up_duration=60,
                test_duration=600,  # 10 minutes
                think_time_min=0.2,
                think_time_max=1.5
            ),
            
            # Heavy load test
            LoadTestConfig(
                target_url="http://api-gateway:8080",
                concurrent_users=200,
                requests_per_user=200,
                ramp_up_duration=120,
                test_duration=1800,  # 30 minutes
                think_time_min=0.1,
                think_time_max=1.0
            ),
            
            # Stress test
            LoadTestConfig(
                target_url="http://api-gateway:8080",
                concurrent_users=500,
                requests_per_user=100,
                ramp_up_duration=300,
                test_duration=3600,  # 60 minutes
                think_time_min=0.05,
                think_time_max=0.5
            )
        ]

    async def run_all_tests(self) -> Dict[str, LoadTestSummary]:
        """Run all load test configurations"""
        results = {}
        
        for i, config in enumerate(self.test_configs):
            test_name = f"load_test_{i+1}_{config.concurrent_users}_users"
            logger.info(f"Starting {test_name}")
            
            tester = APIGatewayLoadTester(config)
            summary = await tester.run_load_test()
            
            # Generate reports
            report_file = f"/tmp/{test_name}_report.html"
            metrics_file = f"/tmp/{test_name}_metrics.json"
            
            tester.generate_report(summary, report_file)
            tester.export_metrics(metrics_file)
            
            results[test_name] = summary
            
            # Wait between tests
            if i < len(self.test_configs) - 1:
                logger.info("Waiting 60 seconds before next test...")
                await asyncio.sleep(60)
        
        return results

# Example usage and testing
async def main():
    # Single load test
    config = LoadTestConfig(
        target_url="http://localhost:8080",
        concurrent_users=20,
        requests_per_user=25,
        ramp_up_duration=30,
        test_duration=300
    )
    
    tester = APIGatewayLoadTester(config)
    summary = await tester.run_load_test()
    
    # Generate reports
    tester.generate_report(summary, "api_gateway_load_test_report.html")
    tester.export_metrics("api_gateway_load_test_metrics.json")
    
    print(f"Load test completed:")
    print(f"  Total requests: {summary.total_requests}")
    print(f"  Success rate: {(1 - summary.error_rate) * 100:.2f}%")
    print(f"  Average response time: {summary.avg_response_time:.3f}s")
    print(f"  95th percentile: {summary.p95_response_time:.3f}s")
    print(f"  Throughput: {summary.requests_per_second:.2f} RPS")

if __name__ == "__main__":
    asyncio.run(main())