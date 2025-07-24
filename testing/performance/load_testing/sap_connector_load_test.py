import asyncio
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
from prometheus_client import Counter, Histogram, Gauge
import aiohttp

logger = logging.getLogger(__name__)

@dataclass
class SAPLoadTestConfig:
    target_url: str
    concurrent_users: int
    requests_per_user: int
    ramp_up_duration: int
    test_duration: int
    think_time_min: float = 0.5
    think_time_max: float = 3.0
    timeout_seconds: int = 60

@dataclass
class SAPTestScenario:
    name: str
    weight: float
    connection_id: str
    operation_id: str
    parameters: Dict[str, Any]
    expected_success: bool = True

@dataclass
class SAPTestResult:
    scenario_name: str
    response_time: float
    success: bool
    error_message: Optional[str] = None
    timestamp: datetime = None
    sap_response_time: Optional[float] = None
    connection_id: str = ""

@dataclass
class SAPLoadTestSummary:
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
    avg_sap_response_time: float
    cpu_usage_avg: float
    memory_usage_avg: float
    connection_performance: Dict[str, Dict[str, float]]

class SAPConnectorLoadTester:
    def __init__(self, config: SAPLoadTestConfig):
        self.config = config
        self.results: List[SAPTestResult] = []
        self.system_metrics: List[Dict[str, float]] = []
        
        # Metrics
        self.request_counter = Counter('sap_load_test_requests_total',
                                     'Total SAP load test requests',
                                     ['scenario', 'connection', 'status'])
        self.response_time_histogram = Histogram('sap_load_test_response_time_seconds',
                                               'SAP response time distribution',
                                               ['scenario', 'connection'])
        self.concurrent_users_gauge = Gauge('sap_load_test_concurrent_users',
                                          'Current concurrent users')
        self.sap_throughput_gauge = Gauge('sap_load_test_throughput_rps',
                                        'SAP requests per second')
        
        # Test scenarios
        self.scenarios = self._create_sap_test_scenarios()
        
        logger.info(f"Initialized SAP load tester with {config.concurrent_users} users")

    def _create_sap_test_scenarios(self) -> List[SAPTestScenario]:
        """Create realistic SAP integration test scenarios"""
        scenarios = [
            # Customer Master Data Scenarios
            SAPTestScenario(
                name="get_customer_data_bapi",
                weight=0.20,
                connection_id="erp_production",
                operation_id="get_customer_data",
                parameters={
                    "customer_number": f"000010000{random.randint(1, 999)}",
                    "company_code": "1000"
                }
            ),
            SAPTestScenario(
                name="get_customer_addresses",
                weight=0.15,
                connection_id="erp_production", 
                operation_id="get_customer_addresses",
                parameters={
                    "customer_number": f"000010000{random.randint(1, 999)}",
                    "address_type": "BILL_TO"
                }
            ),
            
            # Material Master Data Scenarios
            SAPTestScenario(
                name="get_material_info_rfc",
                weight=0.18,
                connection_id="erp_production",
                operation_id="get_material_info",
                parameters={
                    "table": "MARA",
                    "fields": [
                        {"FIELDNAME": "MATNR"},
                        {"FIELDNAME": "MTART"},
                        {"FIELDNAME": "MAKTX"}
                    ],
                    "options": [
                        {"TEXT": f"MATNR EQ '{random.randint(100000, 999999)}'"}
                    ],
                    "rowcount": 100
                }
            ),
            SAPTestScenario(
                name="get_material_prices",
                weight=0.12,
                connection_id="erp_production",
                operation_id="get_material_prices", 
                parameters={
                    "material_number": f"MAT-{random.randint(10000, 99999)}",
                    "plant": "1000",
                    "sales_org": "1000"
                }
            ),
            
            # Sales Order Processing Scenarios
            SAPTestScenario(
                name="create_sales_order",
                weight=0.08,
                connection_id="erp_production",
                operation_id="create_sales_order",
                parameters={
                    "order_header": {
                        "DOC_TYPE": "OR",
                        "SALES_ORG": "1000",
                        "DISTR_CHAN": "10",
                        "DIVISION": "00",
                        "PURCH_NO_C": f"PO-{random.randint(100000, 999999)}"
                    },
                    "order_items": [
                        {
                            "ITM_NUMBER": "000010",
                            "MATERIAL": f"MAT-{random.randint(10000, 99999)}",
                            "REQ_QTY": str(random.randint(1, 100)),
                            "PLANT": "1000"
                        }
                    ],
                    "partners": [
                        {
                            "PARTN_ROLE": "AG",
                            "PARTN_NUMB": f"000010000{random.randint(1, 999)}"
                        }
                    ]
                }
            ),
            SAPTestScenario(
                name="get_sales_order_status",
                weight=0.10,
                connection_id="erp_production",
                operation_id="get_sales_order_status",
                parameters={
                    "sales_document": f"000000{random.randint(1000000, 9999999)}"
                }
            ),
            
            # Purchase Order Scenarios  
            SAPTestScenario(
                name="create_purchase_order",
                weight=0.05,
                connection_id="erp_production",
                operation_id="create_purchase_order",
                parameters={
                    "po_header": {
                        "DOC_TYPE": "NB",
                        "COMP_CODE": "1000",
                        "PURCH_ORG": "1000",
                        "PUR_GROUP": "001",
                        "VENDOR": f"000000{random.randint(100000, 999999)}"
                    },
                    "po_items": [
                        {
                            "PO_ITEM": "00010",
                            "MATERIAL": f"MAT-{random.randint(10000, 99999)}",
                            "QUANTITY": str(random.randint(1, 1000)),
                            "NET_PRICE": str(random.uniform(10.0, 1000.0)),
                            "PLANT": "1000"
                        }
                    ]
                }
            ),
            
            # IDoc Processing Scenarios
            SAPTestScenario(
                name="send_orders_idoc",
                weight=0.04,
                connection_id="erp_production",
                operation_id="send_idoc",
                parameters={
                    "control_record": {
                        "MESTYP": "ORDERS",
                        "IDOCTYP": "ORDERS05",
                        "RCVPRT": "LS",
                        "RCVPRN": "VENDOR_SYSTEM"
                    },
                    "data_records": [
                        {
                            "SEGNAM": "E1EDK01",
                            "BELNR": f"PO-{random.randint(100000, 999999)}",
                            "CURCY": "USD",
                            "WKURS": "1.0"
                        }
                    ]
                }
            ),
            
            # S/4 HANA OData Scenarios
            SAPTestScenario(
                name="odata_products_query",
                weight=0.08,
                connection_id="s4_hana_dev",
                operation_id="odata_products",
                parameters={
                    "filters": f"ProductType eq 'FERT' and ProductGroup eq 'GROUP{random.randint(1, 10)}'",
                    "select": "Product,ProductType,ProductDescription,BaseUnit",
                    "orderby": "Product asc",
                    "top": 20
                }
            )
        ]
        
        return scenarios

    async def run_load_test(self) -> SAPLoadTestSummary:
        """Execute the complete SAP load test"""
        logger.info(f"Starting SAP load test with {self.config.concurrent_users} users")
        
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
                self._simulate_sap_user(user_id, semaphore, delay)
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
        
        logger.info(f"SAP load test completed in {total_duration:.2f} seconds")
        logger.info(f"Total SAP requests: {summary.total_requests}")
        logger.info(f"Success rate: {(1 - summary.error_rate) * 100:.2f}%")
        logger.info(f"Average response time: {summary.avg_response_time:.3f}s")
        logger.info(f"Average SAP response time: {summary.avg_sap_response_time:.3f}s")
        logger.info(f"Throughput: {summary.requests_per_second:.2f} RPS")
        
        return summary

    async def _simulate_sap_user(self, user_id: int, semaphore: asyncio.Semaphore,
                               initial_delay: float):
        """Simulate a single user's SAP integration behavior"""
        await asyncio.sleep(initial_delay)
        
        async with semaphore:
            self.concurrent_users_gauge.inc()
            
            try:
                timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
                
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    for request_num in range(self.config.requests_per_user):
                        # Select scenario based on weights
                        scenario = self._select_scenario()
                        
                        # Execute SAP request
                        result = await self._execute_sap_request(session, scenario, user_id)
                        self.results.append(result)
                        
                        # Update metrics
                        status_label = "success" if result.success else "error"
                        self.request_counter.labels(
                            scenario=scenario.name,
                            connection=scenario.connection_id,
                            status=status_label
                        ).inc()
                        
                        self.response_time_histogram.labels(
                            scenario=scenario.name,
                            connection=scenario.connection_id
                        ).observe(result.response_time)
                        
                        # Think time between requests (SAP operations are typically slower)
                        think_time = random.uniform(
                            self.config.think_time_min,
                            self.config.think_time_max
                        )
                        await asyncio.sleep(think_time)
                        
            except Exception as e:
                logger.error(f"SAP user {user_id} encountered error: {e}")
                
            finally:
                self.concurrent_users_gauge.dec()

    def _select_scenario(self) -> SAPTestScenario:
        """Select a test scenario based on weights"""
        rand = random.random()
        cumulative_weight = 0.0
        
        for scenario in self.scenarios:
            cumulative_weight += scenario.weight
            if rand <= cumulative_weight:
                return scenario
        
        # Fallback to last scenario
        return self.scenarios[-1]

    async def _execute_sap_request(self, session: aiohttp.ClientSession,
                                 scenario: SAPTestScenario, user_id: int) -> SAPTestResult:
        """Execute a single SAP connector request"""
        start_time = time.time()
        
        try:
            # Prepare SAP request payload
            sap_request = {
                "request_id": f"load_test_{user_id}_{int(time.time() * 1000)}",
                "connection_id": scenario.connection_id,
                "operation_id": scenario.operation_id,
                "parameters": scenario.parameters,
                "headers": {
                    "X-Load-Test-User": str(user_id),
                    "X-Load-Test-Scenario": scenario.name
                },
                "priority": 5,
                "timeout_override": 30
            }
            
            # Execute request to SAP connector
            url = f"{self.config.target_url}/execute-request"
            
            async with session.post(
                url=url,
                json=sap_request,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                response_time = time.time() - start_time
                response_data = await response.json()
                
                # Extract SAP-specific timing if available
                sap_response_time = None
                if response_data.get('execution_time_ms'):
                    sap_response_time = response_data['execution_time_ms'] / 1000.0
                
                success = (response.status == 200 and 
                          response_data.get('success', False))
                
                error_message = None
                if not success:
                    error_message = response_data.get('error_message', 
                                                    f"HTTP {response.status}")
                
                return SAPTestResult(
                    scenario_name=scenario.name,
                    response_time=response_time,
                    success=success,
                    error_message=error_message,
                    timestamp=datetime.utcnow(),
                    sap_response_time=sap_response_time,
                    connection_id=scenario.connection_id
                )
                
        except asyncio.TimeoutError:
            return SAPTestResult(
                scenario_name=scenario.name,
                response_time=time.time() - start_time,
                success=False,
                error_message="Request timeout",
                timestamp=datetime.utcnow(),
                connection_id=scenario.connection_id
            )
        except Exception as e:
            return SAPTestResult(
                scenario_name=scenario.name,
                response_time=time.time() - start_time,
                success=False,
                error_message=str(e),
                timestamp=datetime.utcnow(),
                connection_id=scenario.connection_id
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

    def _calculate_summary(self, total_duration: float) -> SAPLoadTestSummary:
        """Calculate test summary statistics"""
        if not self.results:
            return SAPLoadTestSummary(
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
                avg_sap_response_time=0,
                cpu_usage_avg=0,
                memory_usage_avg=0,
                connection_performance={}
            )
        
        # Response time statistics
        response_times = [r.response_time for r in self.results]
        response_times.sort()
        
        # SAP response time statistics
        sap_response_times = [r.sap_response_time for r in self.results 
                             if r.sap_response_time is not None]
        
        total_requests = len(self.results)
        successful_requests = len([r for r in self.results if r.success])
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
        
        # Connection-specific performance
        connection_performance = self._calculate_connection_performance()
        
        return SAPLoadTestSummary(
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
            avg_sap_response_time=statistics.mean(sap_response_times) if sap_response_times else 0,
            cpu_usage_avg=cpu_avg,
            memory_usage_avg=memory_avg,
            connection_performance=connection_performance
        )

    def _calculate_connection_performance(self) -> Dict[str, Dict[str, float]]:
        """Calculate performance metrics per SAP connection"""
        connection_stats = {}
        
        for result in self.results:
            conn_id = result.connection_id
            if conn_id not in connection_stats:
                connection_stats[conn_id] = {
                    'total_requests': 0,
                    'successful_requests': 0,
                    'response_times': [],
                    'sap_response_times': []
                }
            
            stats = connection_stats[conn_id]
            stats['total_requests'] += 1
            if result.success:
                stats['successful_requests'] += 1
            stats['response_times'].append(result.response_time)
            if result.sap_response_time:
                stats['sap_response_times'].append(result.sap_response_time)
        
        # Calculate summary metrics per connection
        performance = {}
        for conn_id, stats in connection_stats.items():
            success_rate = (stats['successful_requests'] / 
                          stats['total_requests'] * 100) if stats['total_requests'] > 0 else 0
            avg_response_time = statistics.mean(stats['response_times']) if stats['response_times'] else 0
            avg_sap_response_time = (statistics.mean(stats['sap_response_times']) 
                                   if stats['sap_response_times'] else 0)
            
            performance[conn_id] = {
                'success_rate': success_rate,
                'avg_response_time': avg_response_time,
                'avg_sap_response_time': avg_sap_response_time,
                'total_requests': stats['total_requests']
            }
        
        return performance

    def generate_report(self, summary: SAPLoadTestSummary, output_file: str):
        """Generate detailed HTML report for SAP load test"""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>SAP Connector Load Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .metric {{ margin: 10px 0; }}
        .metric-value {{ font-weight: bold; color: #2c3e50; }}
        .error {{ color: #e74c3c; }}
        .success {{ color: #27ae60; }}
        .warning {{ color: #f39c12; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .sap-specific {{ background-color: #e8f4f8; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>SAP Connector Load Test Report</h1>
        <p>Generated: {datetime.utcnow().isoformat()}</p>
        <p>Test Configuration: {self.config.concurrent_users} concurrent users, {self.config.requests_per_user} requests per user</p>
        <p class="sap-specific">SAP-specific load testing with enterprise integration patterns</p>
    </div>
    
    <h2>Performance Summary</h2>
    <div class="metric">Total SAP Requests: <span class="metric-value">{summary.total_requests}</span></div>
    <div class="metric">Successful Requests: <span class="metric-value success">{summary.successful_requests}</span></div>
    <div class="metric">Failed Requests: <span class="metric-value error">{summary.failed_requests}</span></div>
    <div class="metric">Success Rate: <span class="metric-value">{(1 - summary.error_rate) * 100:.2f}%</span></div>
    <div class="metric">Average Response Time: <span class="metric-value">{summary.avg_response_time:.3f}s</span></div>
    <div class="metric sap-specific">Average SAP Processing Time: <span class="metric-value">{summary.avg_sap_response_time:.3f}s</span></div>
    <div class="metric">95th Percentile Response Time: <span class="metric-value">{summary.p95_response_time:.3f}s</span></div>
    <div class="metric">99th Percentile Response Time: <span class="metric-value">{summary.p99_response_time:.3f}s</span></div>
    <div class="metric">SAP Requests per Second: <span class="metric-value">{summary.requests_per_second:.2f}</span></div>
    
    <h2>System Resource Usage</h2>
    <div class="metric">Average CPU Usage: <span class="metric-value">{summary.cpu_usage_avg:.1f}%</span></div>
    <div class="metric">Average Memory Usage: <span class="metric-value">{summary.memory_usage_avg:.1f}%</span></div>
    
    <h2>SAP Connection Performance</h2>
    <table>
        <tr>
            <th>Connection ID</th>
            <th>Total Requests</th>
            <th>Success Rate</th>
            <th>Avg Response Time (ms)</th>
            <th>Avg SAP Processing Time (ms)</th>
        </tr>
        {self._generate_connection_performance_table(summary.connection_performance)}
    </table>
    
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
    
    <h2>SAP Scenario Performance</h2>
    <table>
        <tr>
            <th>Scenario</th>
            <th>Requests</th>
            <th>Success Rate</th>
            <th>Avg Response Time (ms)</th>
            <th>Pattern Type</th>
        </tr>
        {self._generate_scenario_performance_table()}
    </table>
    
    <h2>SAP-Specific Recommendations</h2>
    <ul>
        {self._generate_sap_recommendations(summary)}
    </ul>
</body>
</html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"SAP load test report generated: {output_file}")

    def _generate_connection_performance_table(self, 
                                             connection_performance: Dict[str, Dict[str, float]]) -> str:
        """Generate HTML table rows for connection performance"""
        rows = []
        
        for conn_id, metrics in connection_performance.items():
            status_class = "success" if metrics['success_rate'] > 95 else "warning" if metrics['success_rate'] > 80 else "error"
            
            rows.append(f"""
                <tr>
                    <td>{conn_id}</td>
                    <td>{metrics['total_requests']}</td>
                    <td class="{status_class}">{metrics['success_rate']:.1f}%</td>
                    <td>{metrics['avg_response_time'] * 1000:.1f}</td>
                    <td class="sap-specific">{metrics['avg_sap_response_time'] * 1000:.1f}</td>
                </tr>
            """)
        
        return ''.join(rows)

    def _generate_scenario_performance_table(self) -> str:
        """Generate HTML table rows for scenario performance"""
        scenario_stats = {}
        
        for result in self.results:
            if result.scenario_name not in scenario_stats:
                scenario_stats[result.scenario_name] = {
                    'total': 0,
                    'success': 0,
                    'response_times': [],
                    'pattern_type': self._get_scenario_pattern_type(result.scenario_name)
                }
            
            stats = scenario_stats[result.scenario_name]
            stats['total'] += 1
            if result.success:
                stats['success'] += 1
            stats['response_times'].append(result.response_time)
        
        rows = []
        for scenario_name, stats in scenario_stats.items():
            success_rate = (stats['success'] / stats['total'] * 100) if stats['total'] > 0 else 0
            avg_response_time = statistics.mean(stats['response_times']) * 1000 if stats['response_times'] else 0
            status_class = "success" if success_rate > 95 else "warning" if success_rate > 80 else "error"
            
            rows.append(f"""
                <tr>
                    <td>{scenario_name}</td>
                    <td>{stats['total']}</td>
                    <td class="{status_class}">{success_rate:.1f}%</td>
                    <td>{avg_response_time:.1f}</td>
                    <td class="sap-specific">{stats['pattern_type']}</td>
                </tr>
            """)
        
        return ''.join(rows)

    def _get_scenario_pattern_type(self, scenario_name: str) -> str:
        """Determine SAP integration pattern type for scenario"""
        if 'bapi' in scenario_name:
            return 'BAPI'
        elif 'rfc' in scenario_name:
            return 'RFC'
        elif 'idoc' in scenario_name:
            return 'IDoc'
        elif 'odata' in scenario_name:
            return 'OData'
        else:
            return 'REST API'

    def _generate_sap_recommendations(self, summary: SAPLoadTestSummary) -> str:
        """Generate SAP-specific performance recommendations"""
        recommendations = []
        
        if summary.error_rate > 0.05:  # > 5% error rate
            recommendations.append("<li class='error'>High SAP integration error rate detected. Check SAP connection pool configuration and network connectivity.</li>")
        
        if summary.avg_sap_response_time > 5.0:  # > 5 seconds SAP processing
            recommendations.append("<li class='error'>High SAP processing time detected. Consider optimizing BAPI/RFC calls or implementing connection pooling.</li>")
        
        if summary.p95_response_time > 10.0:  # > 10 seconds total
            recommendations.append("<li class='error'>95th percentile response time is very high for SAP operations. Consider implementing caching for frequently accessed data.</li>")
        
        if summary.cpu_usage_avg > 80:
            recommendations.append("<li class='error'>High CPU usage detected. SAP connector may need horizontal scaling or connection pool optimization.</li>")
        
        if summary.requests_per_second < 10:
            recommendations.append("<li class='warning'>Low SAP throughput detected. Consider increasing connection pool size or optimizing SAP queries.</li>")
        
        # SAP-specific recommendations
        for conn_id, metrics in summary.connection_performance.items():
            if metrics['success_rate'] < 90:
                recommendations.append(f"<li class='error'>SAP connection '{conn_id}' has low success rate ({metrics['success_rate']:.1f}%). Check SAP system availability and authentication.</li>")
            
            if metrics['avg_sap_response_time'] > 3.0:
                recommendations.append(f"<li class='warning'>SAP connection '{conn_id}' has high processing time ({metrics['avg_sap_response_time']:.2f}s). Consider query optimization.</li>")
        
        if summary.error_rate < 0.01 and summary.avg_sap_response_time < 2.0:
            recommendations.append("<li class='success'>Excellent SAP integration performance! System is handling SAP load well with low error rate and good response times.</li>")
        
        if not recommendations:
            recommendations.append("<li>SAP integration performance appears normal. Continue monitoring and consider testing with higher load.</li>")
        
        return ''.join(recommendations)

    def export_metrics(self, output_file: str):
        """Export detailed SAP metrics in JSON format"""
        metrics_data = {
            'config': asdict(self.config),
            'results': [asdict(result) for result in self.results],
            'system_metrics': self.system_metrics,
            'scenarios': [asdict(scenario) for scenario in self.scenarios],
            'sap_integration_patterns': {
                scenario.name: self._get_scenario_pattern_type(scenario.name)
                for scenario in self.scenarios
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(metrics_data, f, indent=2, default=str)
        
        logger.info(f"SAP metrics exported to: {output_file}")

# SAP-specific load test runner
class SAPLoadTestRunner:
    def __init__(self):
        self.test_configs = self._create_sap_test_configurations()

    def _create_sap_test_configurations(self) -> List[SAPLoadTestConfig]:
        """Create SAP-specific load test configurations"""
        return [
            # Light SAP load test
            SAPLoadTestConfig(
                target_url="http://sap-connector:8080",
                concurrent_users=5,
                requests_per_user=20,
                ramp_up_duration=60,
                test_duration=300,  # 5 minutes
                think_time_min=1.0,
                think_time_max=3.0,
                timeout_seconds=60
            ),
            
            # Medium SAP load test
            SAPLoadTestConfig(
                target_url="http://sap-connector:8080",
                concurrent_users=15,
                requests_per_user=50,
                ramp_up_duration=120,
                test_duration=900,  # 15 minutes
                think_time_min=0.5,
                think_time_max=2.0,
                timeout_seconds=90
            ),
            
            # Heavy SAP load test
            SAPLoadTestConfig(
                target_url="http://sap-connector:8080",
                concurrent_users=30,
                requests_per_user=100,
                ramp_up_duration=300,
                test_duration=1800,  # 30 minutes
                think_time_min=0.3,
                think_time_max=1.5,
                timeout_seconds=120
            ),
            
            # SAP stress test
            SAPLoadTestConfig(
                target_url="http://sap-connector:8080",
                concurrent_users=50,
                requests_per_user=200,
                ramp_up_duration=600,
                test_duration=3600,  # 60 minutes
                think_time_min=0.2,
                think_time_max=1.0,
                timeout_seconds=180
            )
        ]

    async def run_all_sap_tests(self) -> Dict[str, SAPLoadTestSummary]:
        """Run all SAP load test configurations"""
        results = {}
        
        for i, config in enumerate(self.test_configs):
            test_name = f"sap_load_test_{i+1}_{config.concurrent_users}_users"
            logger.info(f"Starting SAP {test_name}")
            
            tester = SAPConnectorLoadTester(config)
            summary = await tester.run_load_test()
            
            # Generate reports
            report_file = f"/tmp/{test_name}_report.html"
            metrics_file = f"/tmp/{test_name}_metrics.json"
            
            tester.generate_report(summary, report_file)
            tester.export_metrics(metrics_file)
            
            results[test_name] = summary
            
            # Wait between tests (SAP systems need recovery time)
            if i < len(self.test_configs) - 1:
                logger.info("Waiting 120 seconds before next SAP test...")
                await asyncio.sleep(120)
        
        return results

# Example usage and testing
async def main():
    # Single SAP load test
    config = SAPLoadTestConfig(
        target_url="http://localhost:8080",
        concurrent_users=10,
        requests_per_user=20,
        ramp_up_duration=60,
        test_duration=300,
        think_time_min=1.0,
        think_time_max=3.0,
        timeout_seconds=60
    )
    
    tester = SAPConnectorLoadTester(config)
    summary = await tester.run_load_test()
    
    # Generate reports
    tester.generate_report(summary, "sap_connector_load_test_report.html")
    tester.export_metrics("sap_connector_load_test_metrics.json")
    
    print(f"SAP load test completed:")
    print(f"  Total requests: {summary.total_requests}")
    print(f"  Success rate: {(1 - summary.error_rate) * 100:.2f}%")
    print(f"  Average response time: {summary.avg_response_time:.3f}s")
    print(f"  Average SAP processing time: {summary.avg_sap_response_time:.3f}s")
    print(f"  95th percentile: {summary.p95_response_time:.3f}s")
    print(f"  Throughput: {summary.requests_per_second:.2f} RPS")
    
    # Connection performance summary
    print(f"\nSAP Connection Performance:")
    for conn_id, metrics in summary.connection_performance.items():
        print(f"  {conn_id}: {metrics['success_rate']:.1f}% success, {metrics['avg_sap_response_time']:.3f}s avg processing")

if __name__ == "__main__":
    asyncio.run(main())