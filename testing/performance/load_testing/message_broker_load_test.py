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
class MessageBrokerLoadTestConfig:
    target_url: str
    concurrent_producers: int
    concurrent_consumers: int
    messages_per_producer: int
    test_duration: int
    message_size_kb: int = 1
    think_time_min: float = 0.01
    think_time_max: float = 0.1
    timeout_seconds: int = 30

@dataclass
class MessageTestScenario:
    name: str
    weight: float
    topic: str
    priority: str
    message_template: Dict[str, Any]
    delivery_mode: str = "at_least_once"
    delay_seconds: int = 0

@dataclass
class MessageTestResult:
    scenario_name: str
    operation_type: str  # "publish" or "consume"
    response_time: float
    success: bool
    message_id: Optional[str] = None
    error_message: Optional[str] = None
    timestamp: datetime = None
    topic: str = ""
    queue_size_before: Optional[int] = None
    queue_size_after: Optional[int] = None

@dataclass
class MessageBrokerLoadTestSummary:
    total_operations: int
    total_publishes: int
    total_consumes: int
    successful_operations: int
    failed_operations: int
    avg_publish_time: float
    avg_consume_time: float
    p50_publish_time: float
    p95_publish_time: float
    p99_publish_time: float
    max_response_time: float
    min_response_time: float
    messages_per_second: float
    error_rate: float
    avg_queue_depth: float
    cpu_usage_avg: float
    memory_usage_avg: float
    topic_performance: Dict[str, Dict[str, float]]
    throughput_timeline: List[Dict[str, Any]]

class MessageBrokerLoadTester:
    def __init__(self, config: MessageBrokerLoadTestConfig):
        self.config = config
        self.results: List[MessageTestResult] = []
        self.system_metrics: List[Dict[str, float]] = []
        self.throughput_metrics: List[Dict[str, Any]] = []
        
        # Metrics
        self.operation_counter = Counter('message_broker_load_test_operations_total',
                                       'Total message broker operations',
                                       ['scenario', 'operation', 'topic', 'status'])
        self.response_time_histogram = Histogram('message_broker_load_test_response_time_seconds',
                                               'Response time distribution',
                                               ['scenario', 'operation', 'topic'])
        self.concurrent_operations_gauge = Gauge('message_broker_load_test_concurrent_operations',
                                               'Current concurrent operations')
        self.message_throughput_gauge = Gauge('message_broker_load_test_throughput_mps',
                                            'Messages per second')
        self.queue_depth_gauge = Gauge('message_broker_load_test_queue_depth',
                                     'Average queue depth',
                                     ['topic'])
        
        # Test scenarios
        self.scenarios = self._create_message_test_scenarios()
        
        logger.info(f"Initialized message broker load tester with {config.concurrent_producers} producers and {config.concurrent_consumers} consumers")

    def _create_message_test_scenarios(self) -> List[MessageTestScenario]:
        """Create realistic message broker test scenarios"""
        scenarios = [
            # Order Processing Messages
            MessageTestScenario(
                name="order_created",
                weight=0.25,
                topic="orders",
                priority="HIGH",
                message_template={
                    "event_type": "order_created",
                    "order_id": "ORD-{order_id}",
                    "customer_id": "CUST-{customer_id}",
                    "items": [
                        {
                            "product_id": "PROD-{product_id}",
                            "quantity": "{quantity}",
                            "price": "{price}"
                        }
                    ],
                    "total_amount": "{total_amount}",
                    "order_date": "{timestamp}",
                    "shipping_address": {
                        "street": "123 Main St",
                        "city": "Anytown",
                        "state": "CA",
                        "zip": "12345"
                    }
                },
                delivery_mode="exactly_once"
            ),
            MessageTestScenario(
                name="order_updated",
                weight=0.15,
                topic="orders",
                priority="NORMAL",
                message_template={
                    "event_type": "order_updated",
                    "order_id": "ORD-{order_id}",
                    "updates": {
                        "status": "processing",
                        "updated_at": "{timestamp}",
                        "tracking_number": "TRK-{tracking_id}"
                    }
                }
            ),
            
            # Customer Events
            MessageTestScenario(
                name="customer_registered",
                weight=0.10,
                topic="notifications",
                priority="NORMAL",
                message_template={
                    "event_type": "customer_registered",
                    "customer_id": "CUST-{customer_id}",
                    "customer_data": {
                        "name": "Test Customer {customer_id}",
                        "email": "customer{customer_id}@example.com",
                        "registration_date": "{timestamp}",
                        "tier": "standard"
                    },
                    "welcome_actions": ["send_welcome_email", "setup_preferences"]
                }
            ),
            MessageTestScenario(
                name="customer_tier_upgraded",
                weight=0.08,
                topic="notifications",
                priority="HIGH",
                message_template={
                    "event_type": "customer_tier_upgraded",
                    "customer_id": "CUST-{customer_id}",
                    "previous_tier": "standard",
                    "new_tier": "premium",
                    "upgrade_date": "{timestamp}",
                    "benefits_unlocked": ["free_shipping", "priority_support"]
                }
            ),
            
            # Analytics Events
            MessageTestScenario(
                name="page_view_event",
                weight=0.20,
                topic="analytics_events",
                priority="LOW",
                message_template={
                    "event_type": "page_view",
                    "session_id": "SES-{session_id}",
                    "user_id": "USR-{user_id}",
                    "page_url": "/products/category/{category_id}",
                    "timestamp": "{timestamp}",
                    "user_agent": "Mozilla/5.0 (compatible; LoadTest)",
                    "referrer": "https://search.example.com",
                    "viewport": {"width": 1920, "height": 1080}
                }
            ),
            MessageTestScenario(
                name="product_interaction",
                weight=0.12,
                topic="analytics_events",
                priority="NORMAL",
                message_template={
                    "event_type": "product_interaction",
                    "product_id": "PROD-{product_id}",
                    "user_id": "USR-{user_id}",
                    "interaction_type": "view_details",
                    "timestamp": "{timestamp}",
                    "product_category": "electronics",
                    "time_spent_seconds": "{time_spent}"
                }
            ),
            
            # System Alerts
            MessageTestScenario(
                name="system_alert",
                weight=0.05,
                topic="system_alerts",
                priority="CRITICAL",
                message_template={
                    "event_type": "system_alert",
                    "alert_id": "ALT-{alert_id}",
                    "severity": "warning",
                    "service": "payment_processor",
                    "message": "High response time detected",
                    "timestamp": "{timestamp}",
                    "metrics": {
                        "avg_response_time": "{response_time}",
                        "error_rate": "{error_rate}"
                    }
                }
            ),
            
            # Batch Processing Jobs
            MessageTestScenario(
                name="batch_job_request",
                weight=0.05,
                topic="batch_jobs",
                priority="LOW",
                message_template={
                    "event_type": "batch_job_request",
                    "job_id": "JOB-{job_id}",
                    "job_type": "daily_report_generation",
                    "scheduled_time": "{scheduled_time}",
                    "parameters": {
                        "report_date": "{report_date}",
                        "format": "pdf",
                        "recipients": ["analytics@company.com"]
                    }
                },
                delay_seconds=60  # Delayed execution
            )
        ]
        
        return scenarios

    async def run_load_test(self) -> MessageBrokerLoadTestSummary:
        """Execute the complete message broker load test"""
        logger.info(f"Starting message broker load test with {self.config.concurrent_producers} producers and {self.config.concurrent_consumers} consumers")
        
        start_time = time.time()
        
        # Start monitoring tasks
        monitor_task = asyncio.create_task(self._monitor_system_metrics())
        throughput_task = asyncio.create_task(self._monitor_throughput())
        
        # Create producer and consumer tasks
        producer_tasks = []
        consumer_tasks = []
        
        # Create producer semaphore
        producer_semaphore = asyncio.Semaphore(self.config.concurrent_producers)
        
        # Start producers
        for producer_id in range(self.config.concurrent_producers):
            task = asyncio.create_task(
                self._simulate_message_producer(producer_id, producer_semaphore)
            )
            producer_tasks.append(task)
        
        # Start consumers
        consumer_semaphore = asyncio.Semaphore(self.config.concurrent_consumers)
        for consumer_id in range(self.config.concurrent_consumers):
            task = asyncio.create_task(
                self._simulate_message_consumer(consumer_id, consumer_semaphore)
            )
            consumer_tasks.append(task)
        
        # Wait for test duration
        await asyncio.sleep(self.config.test_duration)
        
        # Cancel all tasks
        for task in producer_tasks + consumer_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*producer_tasks, *consumer_tasks, return_exceptions=True)
        
        # Stop monitoring
        monitor_task.cancel()
        throughput_task.cancel()
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Calculate and return summary
        summary = self._calculate_summary(total_duration)
        
        logger.info(f"Message broker load test completed in {total_duration:.2f} seconds")
        logger.info(f"Total operations: {summary.total_operations}")
        logger.info(f"Total publishes: {summary.total_publishes}")
        logger.info(f"Total consumes: {summary.total_consumes}")
        logger.info(f"Success rate: {(1 - summary.error_rate) * 100:.2f}%")
        logger.info(f"Messages per second: {summary.messages_per_second:.2f}")
        logger.info(f"Average queue depth: {summary.avg_queue_depth:.1f}")
        
        return summary

    async def _simulate_message_producer(self, producer_id: int, 
                                       semaphore: asyncio.Semaphore):
        """Simulate a message producer"""
        async with semaphore:
            self.concurrent_operations_gauge.inc()
            
            try:
                timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
                
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    for message_num in range(self.config.messages_per_producer):
                        # Select scenario based on weights
                        scenario = self._select_scenario()
                        
                        # Generate message
                        message = self._generate_message(scenario, producer_id, message_num)
                        
                        # Publish message
                        result = await self._publish_message(session, message, scenario)
                        self.results.append(result)
                        
                        # Update metrics
                        status_label = "success" if result.success else "error"
                        self.operation_counter.labels(
                            scenario=scenario.name,
                            operation="publish",
                            topic=scenario.topic,
                            status=status_label
                        ).inc()
                        
                        self.response_time_histogram.labels(
                            scenario=scenario.name,
                            operation="publish",
                            topic=scenario.topic
                        ).observe(result.response_time)
                        
                        # Think time between messages
                        think_time = random.uniform(
                            self.config.think_time_min,
                            self.config.think_time_max
                        )
                        await asyncio.sleep(think_time)
                        
            except asyncio.CancelledError:
                logger.info(f"Producer {producer_id} cancelled")
            except Exception as e:
                logger.error(f"Producer {producer_id} encountered error: {e}")
                
            finally:
                self.concurrent_operations_gauge.dec()

    async def _simulate_message_consumer(self, consumer_id: int,
                                       semaphore: asyncio.Semaphore):
        """Simulate a message consumer"""
        async with semaphore:
            self.concurrent_operations_gauge.inc()
            
            try:
                timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
                
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    while True:
                        # Try to consume messages from different topics
                        topics = ["orders", "notifications", "analytics_events", "system_alerts", "batch_jobs"]
                        
                        for topic in topics:
                            result = await self._consume_message(session, topic, consumer_id)
                            if result:
                                self.results.append(result)
                                
                                # Update metrics
                                status_label = "success" if result.success else "error"
                                self.operation_counter.labels(
                                    scenario="consume",
                                    operation="consume",
                                    topic=topic,
                                    status=status_label
                                ).inc()
                                
                                self.response_time_histogram.labels(
                                    scenario="consume",
                                    operation="consume",
                                    topic=topic
                                ).observe(result.response_time)
                        
                        # Think time between consumption attempts
                        await asyncio.sleep(random.uniform(0.1, 0.5))
                        
            except asyncio.CancelledError:
                logger.info(f"Consumer {consumer_id} cancelled")
            except Exception as e:
                logger.error(f"Consumer {consumer_id} encountered error: {e}")
                
            finally:
                self.concurrent_operations_gauge.dec()

    def _select_scenario(self) -> MessageTestScenario:
        """Select a test scenario based on weights"""
        rand = random.random()
        cumulative_weight = 0.0
        
        for scenario in self.scenarios:
            cumulative_weight += scenario.weight
            if rand <= cumulative_weight:
                return scenario
        
        # Fallback to last scenario
        return self.scenarios[-1]

    def _generate_message(self, scenario: MessageTestScenario, 
                         producer_id: int, message_num: int) -> Dict[str, Any]:
        """Generate a message based on scenario template"""
        # Generate dynamic values
        values = {
            'order_id': f"{random.randint(100000, 999999)}",
            'customer_id': f"{random.randint(10000, 99999)}",
            'product_id': f"{random.randint(1000, 9999)}",
            'quantity': f"{random.randint(1, 10)}",
            'price': f"{random.uniform(10.0, 500.0):.2f}",
            'total_amount': f"{random.uniform(20.0, 1000.0):.2f}",
            'timestamp': datetime.utcnow().isoformat(),
            'session_id': f"{random.randint(1000000, 9999999)}",
            'user_id': f"{random.randint(10000, 99999)}",
            'category_id': f"{random.randint(1, 20)}",
            'time_spent': f"{random.randint(5, 300)}",
            'alert_id': f"{random.randint(100000, 999999)}",
            'response_time': f"{random.uniform(0.1, 5.0):.3f}",
            'error_rate': f"{random.uniform(0.0, 0.1):.3f}",
            'job_id': f"{random.randint(100000, 999999)}",
            'scheduled_time': (datetime.utcnow() + timedelta(minutes=random.randint(1, 60))).isoformat(),
            'report_date': datetime.utcnow().strftime('%Y-%m-%d'),
            'tracking_id': f"{random.randint(1000000000, 9999999999)}"
        }
        
        # Replace placeholders in template
        message_data = json.loads(json.dumps(scenario.message_template).format(**values))
        
        # Add padding to reach desired message size
        if self.config.message_size_kb > 1:
            padding_size = (self.config.message_size_kb * 1024) - len(json.dumps(message_data))
            if padding_size > 0:
                message_data['_padding'] = 'X' * padding_size
        
        # Create full message
        message = {
            "message_id": f"load_test_{producer_id}_{message_num}_{int(time.time() * 1000)}",
            "topic": scenario.topic,
            "payload": message_data,
            "headers": {
                "source": "load_test",
                "producer_id": str(producer_id),
                "scenario": scenario.name,
                "test_timestamp": datetime.utcnow().isoformat()
            },
            "priority": scenario.priority,
            "delivery_mode": scenario.delivery_mode,
            "delay_seconds": scenario.delay_seconds,
            "correlation_id": f"test_corr_{producer_id}_{message_num}"
        }
        
        return message

    async def _publish_message(self, session: aiohttp.ClientSession, 
                             message: Dict[str, Any], 
                             scenario: MessageTestScenario) -> MessageTestResult:
        """Publish a message to the broker"""
        start_time = time.time()
        
        try:
            # Get queue size before publishing
            queue_size_before = await self._get_queue_size(session, scenario.topic)
            
            url = f"{self.config.target_url}/publish-message"
            
            async with session.post(
                url=url,
                json=message,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                response_time = time.time() - start_time
                response_data = await response.json()
                
                # Get queue size after publishing
                queue_size_after = await self._get_queue_size(session, scenario.topic)
                
                success = response.status == 200 and response_data.get('success', False)
                
                return MessageTestResult(
                    scenario_name=scenario.name,
                    operation_type="publish",
                    response_time=response_time,
                    success=success,
                    message_id=message['message_id'],
                    error_message=None if success else response_data.get('error', f"HTTP {response.status}"),
                    timestamp=datetime.utcnow(),
                    topic=scenario.topic,
                    queue_size_before=queue_size_before,
                    queue_size_after=queue_size_after
                )
                
        except asyncio.TimeoutError:
            return MessageTestResult(
                scenario_name=scenario.name,
                operation_type="publish",
                response_time=time.time() - start_time,
                success=False,
                message_id=message['message_id'],
                error_message="Request timeout",
                timestamp=datetime.utcnow(),
                topic=scenario.topic
            )
        except Exception as e:
            return MessageTestResult(
                scenario_name=scenario.name,
                operation_type="publish",
                response_time=time.time() - start_time,
                success=False,
                message_id=message['message_id'],
                error_message=str(e),
                timestamp=datetime.utcnow(),
                topic=scenario.topic
            )

    async def _consume_message(self, session: aiohttp.ClientSession, 
                             topic: str, consumer_id: int) -> Optional[MessageTestResult]:
        """Consume a message from the broker"""
        start_time = time.time()
        
        try:
            # Get queue size before consuming
            queue_size_before = await self._get_queue_size(session, topic)
            
            if queue_size_before == 0:
                return None  # No messages to consume
            
            url = f"{self.config.target_url}/consume-messages"
            consume_request = {
                "consumer_id": f"load_test_consumer_{consumer_id}",
                "topics": [topic],
                "max_messages": 1
            }
            
            async with session.post(
                url=url,
                json=consume_request,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                response_time = time.time() - start_time
                response_data = await response.json()
                
                # Get queue size after consuming
                queue_size_after = await self._get_queue_size(session, topic)
                
                success = response.status == 200 and len(response_data.get('messages', [])) > 0
                
                message_id = None
                if success and response_data.get('messages'):
                    message_id = response_data['messages'][0].get('message_id')
                    
                    # Acknowledge the message
                    await self._acknowledge_message(session, message_id, consumer_id)
                
                return MessageTestResult(
                    scenario_name="consume",
                    operation_type="consume",
                    response_time=response_time,
                    success=success,
                    message_id=message_id,
                    error_message=None if success else response_data.get('error', f"HTTP {response.status}"),
                    timestamp=datetime.utcnow(),
                    topic=topic,
                    queue_size_before=queue_size_before,
                    queue_size_after=queue_size_after
                )
                
        except Exception as e:
            return MessageTestResult(
                scenario_name="consume",
                operation_type="consume",
                response_time=time.time() - start_time,
                success=False,
                error_message=str(e),
                timestamp=datetime.utcnow(),
                topic=topic
            )

    async def _acknowledge_message(self, session: aiohttp.ClientSession, 
                                 message_id: str, consumer_id: int):
        """Acknowledge message processing"""
        try:
            url = f"{self.config.target_url}/acknowledge-message"
            ack_request = {
                "message_id": message_id,
                "consumer_id": f"load_test_consumer_{consumer_id}"
            }
            
            async with session.post(url=url, json=ack_request) as response:
                pass  # Fire and forget
                
        except Exception as e:
            logger.debug(f"Failed to acknowledge message {message_id}: {e}")

    async def _get_queue_size(self, session: aiohttp.ClientSession, 
                            topic: str) -> int:
        """Get current queue size for a topic"""
        try:
            url = f"{self.config.target_url}/broker-health"
            
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    queues = data.get('queues', {})
                    queue_info = queues.get(topic, {})
                    return queue_info.get('size', 0)
                    
        except Exception as e:
            logger.debug(f"Failed to get queue size for {topic}: {e}")
        
        return 0

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

    async def _monitor_throughput(self):
        """Monitor message throughput over time"""
        last_count = 0
        
        while True:
            try:
                current_count = len(self.results)
                current_time = time.time()
                
                throughput_sample = {
                    'timestamp': current_time,
                    'total_operations': current_count,
                    'operations_since_last': current_count - last_count,
                    'operations_per_second': (current_count - last_count) / 10.0  # 10 second intervals
                }
                
                self.throughput_metrics.append(throughput_sample)
                last_count = current_count
                
                # Update gauge
                self.message_throughput_gauge.set(throughput_sample['operations_per_second'])
                
                await asyncio.sleep(10)  # Sample every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error monitoring throughput: {e}")
                await asyncio.sleep(10)

    def _calculate_summary(self, total_duration: float) -> MessageBrokerLoadTestSummary:
        """Calculate test summary statistics"""
        if not self.results:
            return MessageBrokerLoadTestSummary(
                total_operations=0,
                total_publishes=0,
                total_consumes=0,
                successful_operations=0,
                failed_operations=0,
                avg_publish_time=0,
                avg_consume_time=0,
                p50_publish_time=0,
                p95_publish_time=0,
                p99_publish_time=0,
                max_response_time=0,
                min_response_time=0,
                messages_per_second=0,
                error_rate=1.0,
                avg_queue_depth=0,
                cpu_usage_avg=0,
                memory_usage_avg=0,
                topic_performance={},
                throughput_timeline=[]
            )
        
        # Separate publish and consume operations
        publish_results = [r for r in self.results if r.operation_type == "publish"]
        consume_results = [r for r in self.results if r.operation_type == "consume"]
        
        # Response time statistics
        all_response_times = [r.response_time for r in self.results]
        publish_response_times = [r.response_time for r in publish_results]
        consume_response_times = [r.response_time for r in consume_results]
        
        all_response_times.sort()
        publish_response_times.sort()
        
        total_operations = len(self.results)
        total_publishes = len(publish_results)
        total_consumes = len(consume_results)
        successful_operations = len([r for r in self.results if r.success])
        failed_operations = total_operations - successful_operations
        
        # Calculate percentiles
        def percentile(data: List[float], p: float) -> float:
            if not data:
                return 0.0
            index = int(len(data) * p / 100)
            return data[min(index, len(data) - 1)]
        
        # System metrics averages
        cpu_avg = statistics.mean([m['cpu_percent'] for m in self.system_metrics]) if self.system_metrics else 0
        memory_avg = statistics.mean([m['memory_percent'] for m in self.system_metrics]) if self.system_metrics else 0
        
        # Queue depth analysis
        queue_depths = []
        for result in self.results:
            if result.queue_size_before is not None:
                queue_depths.append(result.queue_size_before)
            if result.queue_size_after is not None:
                queue_depths.append(result.queue_size_after)
        
        avg_queue_depth = statistics.mean(queue_depths) if queue_depths else 0
        
        # Topic-specific performance
        topic_performance = self._calculate_topic_performance()
        
        return MessageBrokerLoadTestSummary(
            total_operations=total_operations,
            total_publishes=total_publishes,
            total_consumes=total_consumes,
            successful_operations=successful_operations,
            failed_operations=failed_operations,
            avg_publish_time=statistics.mean(publish_response_times) if publish_response_times else 0,
            avg_consume_time=statistics.mean(consume_response_times) if consume_response_times else 0,
            p50_publish_time=percentile(publish_response_times, 50),
            p95_publish_time=percentile(publish_response_times, 95),
            p99_publish_time=percentile(publish_response_times, 99),
            max_response_time=max(all_response_times) if all_response_times else 0,
            min_response_time=min(all_response_times) if all_response_times else 0,
            messages_per_second=total_operations / total_duration,
            error_rate=failed_operations / total_operations if total_operations > 0 else 0,
            avg_queue_depth=avg_queue_depth,
            cpu_usage_avg=cpu_avg,
            memory_usage_avg=memory_avg,
            topic_performance=topic_performance,
            throughput_timeline=self.throughput_metrics
        )

    def _calculate_topic_performance(self) -> Dict[str, Dict[str, float]]:
        """Calculate performance metrics per topic"""
        topic_stats = {}
        
        for result in self.results:
            topic = result.topic
            if topic not in topic_stats:
                topic_stats[topic] = {
                    'total_operations': 0,
                    'successful_operations': 0,
                    'publish_operations': 0,
                    'consume_operations': 0,
                    'response_times': []
                }
            
            stats = topic_stats[topic]
            stats['total_operations'] += 1
            if result.success:
                stats['successful_operations'] += 1
            if result.operation_type == "publish":
                stats['publish_operations'] += 1
            else:
                stats['consume_operations'] += 1
            stats['response_times'].append(result.response_time)
        
        # Calculate summary metrics per topic
        performance = {}
        for topic, stats in topic_stats.items():
            success_rate = (stats['successful_operations'] / 
                          stats['total_operations'] * 100) if stats['total_operations'] > 0 else 0
            avg_response_time = statistics.mean(stats['response_times']) if stats['response_times'] else 0
            
            performance[topic] = {
                'success_rate': success_rate,
                'avg_response_time': avg_response_time,
                'total_operations': stats['total_operations'],
                'publish_operations': stats['publish_operations'],
                'consume_operations': stats['consume_operations']
            }
        
        return performance

    def generate_report(self, summary: MessageBrokerLoadTestSummary, output_file: str):
        """Generate detailed HTML report for message broker load test"""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Message Broker Load Test Report</title>
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
        .messaging-specific {{ background-color: #e8f4f8; }}
        .chart {{ margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Message Broker Load Test Report</h1>
        <p>Generated: {datetime.utcnow().isoformat()}</p>
        <p>Test Configuration: {self.config.concurrent_producers} producers, {self.config.concurrent_consumers} consumers, {self.config.messages_per_producer} messages per producer</p>
        <p class="messaging-specific">Message size: {self.config.message_size_kb}KB, Test duration: {self.config.test_duration}s</p>
    </div>
    
    <h2>Performance Summary</h2>
    <div class="metric">Total Operations: <span class="metric-value">{summary.total_operations}</span></div>
    <div class="metric">Total Publishes: <span class="metric-value">{summary.total_publishes}</span></div>
    <div class="metric">Total Consumes: <span class="metric-value">{summary.total_consumes}</span></div>
    <div class="metric">Successful Operations: <span class="metric-value success">{summary.successful_operations}</span></div>
    <div class="metric">Failed Operations: <span class="metric-value error">{summary.failed_operations}</span></div>
    <div class="metric">Success Rate: <span class="metric-value">{(1 - summary.error_rate) * 100:.2f}%</span></div>
    <div class="metric messaging-specific">Messages per Second: <span class="metric-value">{summary.messages_per_second:.2f}</span></div>
    <div class="metric">Average Publish Time: <span class="metric-value">{summary.avg_publish_time:.3f}s</span></div>
    <div class="metric">Average Consume Time: <span class="metric-value">{summary.avg_consume_time:.3f}s</span></div>
    <div class="metric messaging-specific">Average Queue Depth: <span class="metric-value">{summary.avg_queue_depth:.1f}</span></div>
    
    <h2>System Resource Usage</h2>
    <div class="metric">Average CPU Usage: <span class="metric-value">{summary.cpu_usage_avg:.1f}%</span></div>
    <div class="metric">Average Memory Usage: <span class="metric-value">{summary.memory_usage_avg:.1f}%</span></div>
    
    <h2>Response Time Distribution (Publish Operations)</h2>
    <table>
        <tr>
            <th>Percentile</th>
            <th>Response Time (ms)</th>
        </tr>
        <tr><td>50th (Median)</td><td>{summary.p50_publish_time * 1000:.1f}</td></tr>
        <tr><td>95th</td><td>{summary.p95_publish_time * 1000:.1f}</td></tr>
        <tr><td>99th</td><td>{summary.p99_publish_time * 1000:.1f}</td></tr>
        <tr><td>Maximum</td><td>{summary.max_response_time * 1000:.1f}</td></tr>
        <tr><td>Minimum</td><td>{summary.min_response_time * 1000:.1f}</td></tr>
    </table>
    
    <h2>Topic Performance</h2>
    <table>
        <tr>
            <th>Topic</th>
            <th>Total Operations</th>
            <th>Publishes</th>
            <th>Consumes</th>
            <th>Success Rate</th>
            <th>Avg Response Time (ms)</th>
        </tr>
        {self._generate_topic_performance_table(summary.topic_performance)}
    </table>
    
    <h2>Message Broker Recommendations</h2>
    <ul>
        {self._generate_message_broker_recommendations(summary)}
    </ul>
</body>
</html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Message broker load test report generated: {output_file}")

    def _generate_topic_performance_table(self, topic_performance: Dict[str, Dict[str, float]]) -> str:
        """Generate HTML table rows for topic performance"""
        rows = []
        
        for topic, metrics in topic_performance.items():
            status_class = "success" if metrics['success_rate'] > 95 else "warning" if metrics['success_rate'] > 80 else "error"
            
            rows.append(f"""
                <tr>
                    <td class="messaging-specific">{topic}</td>
                    <td>{metrics['total_operations']}</td>
                    <td>{metrics['publish_operations']}</td>
                    <td>{metrics['consume_operations']}</td>
                    <td class="{status_class}">{metrics['success_rate']:.1f}%</td>
                    <td>{metrics['avg_response_time'] * 1000:.1f}</td>
                </tr>
            """)
        
        return ''.join(rows)

    def _generate_message_broker_recommendations(self, summary: MessageBrokerLoadTestSummary) -> str:
        """Generate message broker specific recommendations"""
        recommendations = []
        
        if summary.error_rate > 0.05:  # > 5% error rate
            recommendations.append("<li class='error'>High message broker error rate detected. Check queue capacity and consumer processing capability.</li>")
        
        if summary.avg_publish_time > 0.1:  # > 100ms publish time
            recommendations.append("<li class='warning'>High message publish latency detected. Consider optimizing message serialization or broker configuration.</li>")
        
        if summary.avg_consume_time > 0.2:  # > 200ms consume time
            recommendations.append("<li class='warning'>High message consume latency detected. Optimize consumer processing logic or increase consumer count.</li>")
        
        if summary.avg_queue_depth > 1000:
            recommendations.append("<li class='error'>High average queue depth indicates consumer lag. Consider scaling consumers or optimizing processing.</li>")
        
        if summary.messages_per_second < 100:
            recommendations.append("<li class='warning'>Low message throughput detected. Consider optimizing broker configuration or increasing parallelism.</li>")
        
        # Topic-specific recommendations
        for topic, metrics in summary.topic_performance.items():
            if metrics['success_rate'] < 90:
                recommendations.append(f"<li class='error'>Topic '{topic}' has low success rate ({metrics['success_rate']:.1f}%). Investigate topic-specific issues.</li>")
            
            if metrics['consume_operations'] == 0 and metrics['publish_operations'] > 0:
                recommendations.append(f"<li class='warning'>Topic '{topic}' has messages being published but not consumed. Check consumer configuration.</li>")
        
        if summary.error_rate < 0.01 and summary.messages_per_second > 1000:
            recommendations.append("<li class='success'>Excellent message broker performance! High throughput with low error rate.</li>")
        
        if not recommendations:
            recommendations.append("<li>Message broker performance appears normal. Continue monitoring and consider testing with higher message volume.</li>")
        
        return ''.join(recommendations)

    def export_metrics(self, output_file: str):
        """Export detailed message broker metrics in JSON format"""
        metrics_data = {
            'config': asdict(self.config),
            'results': [asdict(result) for result in self.results],
            'system_metrics': self.system_metrics,
            'throughput_timeline': self.throughput_metrics,
            'scenarios': [asdict(scenario) for scenario in self.scenarios]
        }
        
        with open(output_file, 'w') as f:
            json.dump(metrics_data, f, indent=2, default=str)
        
        logger.info(f"Message broker metrics exported to: {output_file}")

# Example usage and testing
async def main():
    # Single message broker load test
    config = MessageBrokerLoadTestConfig(
        target_url="http://localhost:8080",
        concurrent_producers=10,
        concurrent_consumers=5,
        messages_per_producer=100,
        test_duration=300,  # 5 minutes
        message_size_kb=2,
        think_time_min=0.01,
        think_time_max=0.1
    )
    
    tester = MessageBrokerLoadTester(config)
    summary = await tester.run_load_test()
    
    # Generate reports
    tester.generate_report(summary, "message_broker_load_test_report.html")
    tester.export_metrics("message_broker_load_test_metrics.json")
    
    print(f"Message broker load test completed:")
    print(f"  Total operations: {summary.total_operations}")
    print(f"  Total publishes: {summary.total_publishes}")
    print(f"  Total consumes: {summary.total_consumes}")
    print(f"  Success rate: {(1 - summary.error_rate) * 100:.2f}%")
    print(f"  Messages per second: {summary.messages_per_second:.2f}")
    print(f"  Average queue depth: {summary.avg_queue_depth:.1f}")
    print(f"  Average publish time: {summary.avg_publish_time:.3f}s")
    print(f"  Average consume time: {summary.avg_consume_time:.3f}s")
    
    # Topic performance summary
    print(f"\nTopic Performance:")
    for topic, metrics in summary.topic_performance.items():
        print(f"  {topic}: {metrics['success_rate']:.1f}% success, {metrics['total_operations']} ops")

if __name__ == "__main__":
    asyncio.run(main())