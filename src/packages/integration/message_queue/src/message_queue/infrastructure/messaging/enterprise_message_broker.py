import asyncio
import logging
import json
import time
import hashlib
from typing import Dict, List, Optional, Any, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import aioredis
import aiokafka
from prometheus_client import Counter, Histogram, Gauge, Summary
import pickle

logger = logging.getLogger(__name__)

class MessagePriority(Enum):
    LOW = 1
    NORMAL = 5
    HIGH = 10
    CRITICAL = 15

class DeliveryMode(Enum):
    AT_MOST_ONCE = "at_most_once"
    AT_LEAST_ONCE = "at_least_once"
    EXACTLY_ONCE = "exactly_once"

class QueueType(Enum):
    STANDARD = "standard"
    FIFO = "fifo"
    PRIORITY = "priority"
    DELAYED = "delayed"
    DEAD_LETTER = "dead_letter"

class MessageStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    DEAD_LETTER = "dead_letter"

@dataclass
class Message:
    message_id: str
    topic: str
    payload: Any
    headers: Dict[str, Any]
    priority: MessagePriority = MessagePriority.NORMAL
    delivery_mode: DeliveryMode = DeliveryMode.AT_LEAST_ONCE
    timestamp: datetime = None
    expiry_time: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    delay_seconds: int = 0
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    content_type: str = "application/json"
    compression: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

@dataclass
class Queue:
    name: str
    queue_type: QueueType
    max_size: int = 10000
    ttl_seconds: Optional[int] = None
    dead_letter_queue: Optional[str] = None
    max_delivery_attempts: int = 3
    visibility_timeout: int = 30
    batch_size: int = 10
    auto_acknowledge: bool = False

@dataclass
class Consumer:
    consumer_id: str
    name: str
    topics: List[str]
    group_id: str
    handler: Callable
    batch_processing: bool = False
    max_batch_size: int = 10
    processing_timeout: int = 30
    auto_commit: bool = True
    offset_reset: str = "earliest"

@dataclass
class ConsumerGroup:
    group_id: str
    consumers: List[Consumer]
    partition_assignment: Dict[str, List[int]]
    offset_management: Dict[str, int]
    rebalance_strategy: str = "range"

class EnterpriseMessageBroker:
    def __init__(self):
        self.queues: Dict[str, Queue] = {}
        self.consumers: Dict[str, Consumer] = {}
        self.consumer_groups: Dict[str, ConsumerGroup] = {}
        self.message_store: Dict[str, Dict[str, Message]] = defaultdict(dict)
        self.priority_queues: Dict[str, List[deque]] = {}
        self.delayed_messages: Dict[str, List[Message]] = defaultdict(list)
        self.processing_messages: Dict[str, Message] = {}
        self.dead_letter_messages: Dict[str, List[Message]] = defaultdict(list)
        
        # External integrations
        self.kafka_producer: Optional[aiokafka.AIOKafkaProducer] = None
        self.kafka_consumer: Optional[aiokafka.AIOKafkaConsumer] = None
        self.redis_client: Optional[aioredis.Redis] = None
        
        # Message routing and filtering
        self.routing_rules: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.message_filters: Dict[str, List[Callable]] = defaultdict(list)
        
        # Metrics
        self.messages_produced = Counter('message_broker_messages_produced_total',
                                       'Total messages produced',
                                       ['topic', 'priority'])
        self.messages_consumed = Counter('message_broker_messages_consumed_total',
                                       'Total messages consumed',
                                       ['topic', 'consumer_group', 'status'])
        self.message_processing_time = Histogram('message_broker_processing_duration_seconds',
                                               'Message processing duration',
                                               ['topic', 'consumer_group'])
        self.queue_size = Gauge('message_broker_queue_size',
                              'Current queue size',
                              ['queue_name', 'queue_type'])
        self.consumer_lag = Gauge('message_broker_consumer_lag',
                                'Consumer lag',
                                ['topic', 'consumer_group'])
        self.active_consumers = Gauge('message_broker_active_consumers',
                                    'Number of active consumers',
                                    ['consumer_group'])
        
        # Background tasks
        self.background_tasks: Set[asyncio.Task] = set()
        self.running = False
        
        logger.info("Enterprise Message Broker initialized")

    async def initialize(self):
        """Initialize the message broker"""
        try:
            # Initialize Redis for message persistence and coordination
            self.redis_client = aioredis.from_url(
                "redis://localhost:6379/9",
                decode_responses=False  # Keep binary for message serialization
            )
            
            # Initialize Kafka producer
            self.kafka_producer = aiokafka.AIOKafkaProducer(
                bootstrap_servers=['localhost:9092'],
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                compression_type='gzip',
                acks='all',
                retries=3,
                batch_size=16384,
                linger_ms=5
            )
            await self.kafka_producer.start()
            
            # Load default queues
            await self._load_default_queues()
            
            # Start background tasks
            await self._start_background_tasks()
            
            self.running = True
            logger.info("Message Broker initialization completed")
            
        except Exception as e:
            logger.error(f"Failed to initialize Message Broker: {e}")
            raise

    async def _load_default_queues(self):
        """Load default queue configurations"""
        default_queues = [
            Queue(
                name="orders",
                queue_type=QueueType.FIFO,
                max_size=50000,
                ttl_seconds=86400,  # 24 hours
                dead_letter_queue="orders_dlq",
                max_delivery_attempts=5
            ),
            Queue(
                name="notifications",
                queue_type=QueueType.PRIORITY,
                max_size=100000,
                ttl_seconds=3600,  # 1 hour
                dead_letter_queue="notifications_dlq",
                max_delivery_attempts=3
            ),
            Queue(
                name="analytics_events",
                queue_type=QueueType.STANDARD,
                max_size=1000000,
                ttl_seconds=604800,  # 7 days
                dead_letter_queue="analytics_dlq",
                max_delivery_attempts=2
            ),
            Queue(
                name="system_alerts",
                queue_type=QueueType.PRIORITY,
                max_size=10000,
                ttl_seconds=7200,  # 2 hours
                dead_letter_queue="alerts_dlq",
                max_delivery_attempts=10
            ),
            Queue(
                name="batch_jobs",
                queue_type=QueueType.DELAYED,
                max_size=25000,
                ttl_seconds=259200,  # 3 days
                dead_letter_queue="batch_jobs_dlq",
                max_delivery_attempts=5
            )
        ]
        
        for queue in default_queues:
            await self.create_queue(queue)
        
        # Create dead letter queues
        dlq_names = ["orders_dlq", "notifications_dlq", "analytics_dlq", "alerts_dlq", "batch_jobs_dlq"]
        for dlq_name in dlq_names:
            dlq = Queue(
                name=dlq_name,
                queue_type=QueueType.DEAD_LETTER,
                max_size=10000,
                ttl_seconds=2592000  # 30 days
            )
            await self.create_queue(dlq)
        
        logger.info(f"Loaded {len(default_queues)} default queues")

    async def _start_background_tasks(self):
        """Start background processing tasks"""
        tasks = [
            self._process_delayed_messages(),
            self._cleanup_expired_messages(),
            self._update_metrics(),
            self._rebalance_consumers(),
            self._process_dead_letter_messages()
        ]
        
        for task_coro in tasks:
            task = asyncio.create_task(task_coro)
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)

    async def create_queue(self, queue: Queue) -> bool:
        """Create a new message queue"""
        try:
            self.queues[queue.name] = queue
            
            # Initialize priority queues for priority queue type
            if queue.queue_type == QueueType.PRIORITY:
                self.priority_queues[queue.name] = [
                    deque() for _ in range(16)  # 16 priority levels (0-15)
                ]
            
            # Update metrics
            self.queue_size.labels(
                queue_name=queue.name,
                queue_type=queue.queue_type.value
            ).set(0)
            
            logger.info(f"Created queue: {queue.name} ({queue.queue_type.value})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create queue {queue.name}: {e}")
            return False

    async def publish_message(self, message: Message) -> bool:
        """Publish a message to a topic/queue"""
        try:
            # Validate queue exists
            queue = self.queues.get(message.topic)
            if not queue:
                logger.error(f"Queue not found: {message.topic}")
                return False
            
            # Check queue size limits
            current_size = await self._get_queue_size(message.topic)
            if current_size >= queue.max_size:
                logger.warning(f"Queue {message.topic} is full")
                return False
            
            # Generate message ID if not provided
            if not message.message_id:
                message.message_id = self._generate_message_id(message)
            
            # Set expiry time if queue has TTL
            if queue.ttl_seconds and not message.expiry_time:
                message.expiry_time = datetime.utcnow() + timedelta(seconds=queue.ttl_seconds)
            
            # Handle delayed messages
            if message.delay_seconds > 0:
                scheduled_time = datetime.utcnow() + timedelta(seconds=message.delay_seconds)
                message.timestamp = scheduled_time
                self.delayed_messages[message.topic].append(message)
                
                # Persist to Redis
                await self._persist_delayed_message(message)
                
                logger.debug(f"Scheduled delayed message {message.message_id} for {scheduled_time}")
                return True
            
            # Route message to appropriate queue type
            if queue.queue_type == QueueType.PRIORITY:
                await self._publish_to_priority_queue(message, queue)
            elif queue.queue_type == QueueType.FIFO:
                await self._publish_to_fifo_queue(message, queue)
            else:
                await self._publish_to_standard_queue(message, queue)
            
            # Publish to Kafka for external consumers
            await self._publish_to_kafka(message)
            
            # Persist message
            await self._persist_message(message)
            
            # Update metrics
            self.messages_produced.labels(
                topic=message.topic,
                priority=message.priority.name
            ).inc()
            
            # Update queue size metric
            new_size = await self._get_queue_size(message.topic)
            self.queue_size.labels(
                queue_name=message.topic,
                queue_type=queue.queue_type.value
            ).set(new_size)
            
            logger.debug(f"Published message {message.message_id} to {message.topic}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to publish message: {e}")
            return False

    async def _publish_to_priority_queue(self, message: Message, queue: Queue):
        """Publish message to priority queue"""
        priority_level = min(message.priority.value, 15)  # Cap at 15
        self.priority_queues[message.topic][priority_level].append(message)

    async def _publish_to_fifo_queue(self, message: Message, queue: Queue):
        """Publish message to FIFO queue"""
        if message.topic not in self.message_store:
            self.message_store[message.topic] = {}
        
        # Use timestamp as key to maintain order
        timestamp_key = f"{message.timestamp.timestamp()}_{message.message_id}"
        self.message_store[message.topic][timestamp_key] = message

    async def _publish_to_standard_queue(self, message: Message, queue: Queue):
        """Publish message to standard queue"""
        if message.topic not in self.message_store:
            self.message_store[message.topic] = {}
        
        self.message_store[message.topic][message.message_id] = message

    async def _publish_to_kafka(self, message: Message):
        """Publish message to Kafka for external integration"""
        try:
            if self.kafka_producer:
                message_data = {
                    'message_id': message.message_id,
                    'payload': message.payload,
                    'headers': message.headers,
                    'timestamp': message.timestamp.isoformat(),
                    'priority': message.priority.value,
                    'correlation_id': message.correlation_id
                }
                
                await self.kafka_producer.send(
                    topic=f"enterprise_{message.topic}",
                    key=message.correlation_id,
                    value=message_data,
                    headers=[(k, str(v).encode()) for k, v in message.headers.items()]
                )
                
        except Exception as e:
            logger.error(f"Failed to publish to Kafka: {e}")

    async def register_consumer(self, consumer: Consumer) -> bool:
        """Register a message consumer"""
        try:
            self.consumers[consumer.consumer_id] = consumer
            
            # Create or update consumer group
            if consumer.group_id not in self.consumer_groups:
                self.consumer_groups[consumer.group_id] = ConsumerGroup(
                    group_id=consumer.group_id,
                    consumers=[],
                    partition_assignment={},
                    offset_management={}
                )
            
            group = self.consumer_groups[consumer.group_id]
            if consumer not in group.consumers:
                group.consumers.append(consumer)
            
            # Update metrics
            self.active_consumers.labels(consumer_group=consumer.group_id).set(
                len(group.consumers)
            )
            
            logger.info(f"Registered consumer: {consumer.name} in group {consumer.group_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register consumer {consumer.consumer_id}: {e}")
            return False

    async def consume_messages(self, consumer_id: str) -> Optional[List[Message]]:
        """Consume messages for a specific consumer"""
        try:
            consumer = self.consumers.get(consumer_id)
            if not consumer:
                logger.error(f"Consumer not found: {consumer_id}")
                return None
            
            messages = []
            max_messages = consumer.max_batch_size if consumer.batch_processing else 1
            
            for topic in consumer.topics:
                topic_messages = await self._consume_from_topic(topic, max_messages)
                messages.extend(topic_messages)
                
                if len(messages) >= max_messages:
                    break
            
            # Mark messages as processing
            for message in messages:
                self.processing_messages[message.message_id] = message
            
            # Update metrics
            if messages:
                self.messages_consumed.labels(
                    topic=messages[0].topic,
                    consumer_group=consumer.group_id,
                    status='consumed'
                ).inc(len(messages))
            
            return messages if messages else None
            
        except Exception as e:
            logger.error(f"Failed to consume messages for {consumer_id}: {e}")
            return None

    async def _consume_from_topic(self, topic: str, max_messages: int) -> List[Message]:
        """Consume messages from a specific topic"""
        try:
            queue = self.queues.get(topic)
            if not queue:
                return []
            
            messages = []
            
            if queue.queue_type == QueueType.PRIORITY:
                messages = await self._consume_from_priority_queue(topic, max_messages)
            elif queue.queue_type == QueueType.FIFO:
                messages = await self._consume_from_fifo_queue(topic, max_messages)
            else:
                messages = await self._consume_from_standard_queue(topic, max_messages)
            
            # Filter expired messages
            current_time = datetime.utcnow()
            valid_messages = []
            
            for message in messages:
                if message.expiry_time and current_time > message.expiry_time:
                    logger.debug(f"Message {message.message_id} expired, moving to DLQ")
                    await self._move_to_dead_letter_queue(message, "expired")
                else:
                    valid_messages.append(message)
            
            return valid_messages
            
        except Exception as e:
            logger.error(f"Failed to consume from topic {topic}: {e}")
            return []

    async def _consume_from_priority_queue(self, topic: str, max_messages: int) -> List[Message]:
        """Consume messages from priority queue (highest priority first)"""
        messages = []
        priority_queues = self.priority_queues.get(topic, [])
        
        # Start from highest priority (15) and work down
        for priority_level in reversed(range(16)):
            if len(messages) >= max_messages:
                break
                
            queue = priority_queues[priority_level]
            while queue and len(messages) < max_messages:
                message = queue.popleft()
                messages.append(message)
        
        return messages

    async def _consume_from_fifo_queue(self, topic: str, max_messages: int) -> List[Message]:
        """Consume messages from FIFO queue (oldest first)"""
        messages = []
        topic_messages = self.message_store.get(topic, {})
        
        # Sort by timestamp to maintain FIFO order
        sorted_keys = sorted(topic_messages.keys())
        
        for key in sorted_keys[:max_messages]:
            message = topic_messages.pop(key)
            messages.append(message)
        
        return messages

    async def _consume_from_standard_queue(self, topic: str, max_messages: int) -> List[Message]:
        """Consume messages from standard queue"""
        messages = []
        topic_messages = self.message_store.get(topic, {})
        
        # Get up to max_messages
        keys = list(topic_messages.keys())[:max_messages]
        
        for key in keys:
            message = topic_messages.pop(key)
            messages.append(message)
        
        return messages

    async def acknowledge_message(self, message_id: str, consumer_id: str) -> bool:
        """Acknowledge message processing completion"""
        try:
            # Remove from processing messages
            message = self.processing_messages.pop(message_id, None)
            if not message:
                logger.warning(f"Message {message_id} not found in processing queue")
                return False
            
            consumer = self.consumers.get(consumer_id)
            if consumer:
                # Update metrics
                self.messages_consumed.labels(
                    topic=message.topic,
                    consumer_group=consumer.group_id,
                    status='acknowledged'
                ).inc()
            
            # Remove from Redis persistence
            await self._remove_persisted_message(message)
            
            logger.debug(f"Acknowledged message {message_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to acknowledge message {message_id}: {e}")
            return False

    async def reject_message(self, message_id: str, consumer_id: str, 
                           requeue: bool = True) -> bool:
        """Reject a message and optionally requeue it"""
        try:
            message = self.processing_messages.pop(message_id, None)
            if not message:
                logger.warning(f"Message {message_id} not found in processing queue")
                return False
            
            consumer = self.consumers.get(consumer_id)
            queue = self.queues.get(message.topic)
            
            if not queue:
                logger.error(f"Queue not found: {message.topic}")
                return False
            
            # Increment retry count
            message.retry_count += 1
            
            # Check if max retries exceeded
            if message.retry_count >= message.max_retries:
                logger.warning(f"Message {message_id} exceeded max retries, moving to DLQ")
                await self._move_to_dead_letter_queue(message, "max_retries_exceeded")
            elif requeue:
                # Requeue with exponential backoff delay
                delay_seconds = min(2 ** message.retry_count, 300)  # Max 5 minutes
                message.delay_seconds = delay_seconds
                message.timestamp = datetime.utcnow() + timedelta(seconds=delay_seconds)
                
                self.delayed_messages[message.topic].append(message)
                await self._persist_delayed_message(message)
                
                logger.debug(f"Requeued message {message_id} with {delay_seconds}s delay")
            
            # Update metrics
            if consumer:
                self.messages_consumed.labels(
                    topic=message.topic,
                    consumer_group=consumer.group_id,
                    status='rejected'
                ).inc()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to reject message {message_id}: {e}")
            return False

    async def _move_to_dead_letter_queue(self, message: Message, reason: str):
        """Move message to dead letter queue"""
        try:
            queue = self.queues.get(message.topic)
            if not queue or not queue.dead_letter_queue:
                logger.warning(f"No dead letter queue configured for {message.topic}")
                return
            
            # Add reason to message headers
            message.headers['dlq_reason'] = reason
            message.headers['dlq_timestamp'] = datetime.utcnow().isoformat()
            message.headers['original_topic'] = message.topic
            
            # Move to dead letter queue
            message.topic = queue.dead_letter_queue
            self.dead_letter_messages[queue.dead_letter_queue].append(message)
            
            # Persist to Redis
            await self._persist_message(message)
            
            logger.info(f"Moved message {message.message_id} to DLQ: {reason}")
            
        except Exception as e:
            logger.error(f"Failed to move message to DLQ: {e}")

    async def add_routing_rule(self, topic: str, rule: Dict[str, Any]) -> bool:
        """Add a message routing rule"""
        try:
            self.routing_rules[topic].append(rule)
            logger.info(f"Added routing rule for topic {topic}: {rule}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add routing rule: {e}")
            return False

    async def add_message_filter(self, topic: str, filter_func: Callable) -> bool:
        """Add a message filter function"""
        try:
            self.message_filters[topic].append(filter_func)
            logger.info(f"Added message filter for topic {topic}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add message filter: {e}")
            return False

    async def _process_delayed_messages(self):
        """Background task to process delayed messages"""
        while self.running:
            try:
                current_time = datetime.utcnow()
                
                for topic, messages in self.delayed_messages.items():
                    ready_messages = []
                    remaining_messages = []
                    
                    for message in messages:
                        if message.timestamp <= current_time:
                            ready_messages.append(message)
                        else:
                            remaining_messages.append(message)
                    
                    # Update delayed messages list
                    self.delayed_messages[topic] = remaining_messages
                    
                    # Publish ready messages
                    for message in ready_messages:
                        message.delay_seconds = 0  # Reset delay
                        await self.publish_message(message)
                        await self._remove_delayed_message(message)
                
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"Error processing delayed messages: {e}")
                await asyncio.sleep(5)

    async def _cleanup_expired_messages(self):
        """Background task to clean up expired messages"""
        while self.running:
            try:
                current_time = datetime.utcnow()
                
                # Clean up expired messages from all queues
                for topic, messages in self.message_store.items():
                    expired_keys = []
                    
                    for key, message in messages.items():
                        if message.expiry_time and current_time > message.expiry_time:
                            expired_keys.append(key)
                    
                    # Remove expired messages
                    for key in expired_keys:
                        expired_message = messages.pop(key)
                        logger.debug(f"Expired message {expired_message.message_id} from {topic}")
                        await self._move_to_dead_letter_queue(expired_message, "expired")
                
                # Clean up processing messages that have timed out
                timed_out_messages = []
                for message_id, message in self.processing_messages.items():
                    processing_time = (current_time - message.timestamp).total_seconds()
                    if processing_time > 300:  # 5 minutes timeout
                        timed_out_messages.append(message_id)
                
                for message_id in timed_out_messages:
                    message = self.processing_messages.pop(message_id)
                    logger.warning(f"Processing timeout for message {message_id}")
                    await self.reject_message(message_id, "system", requeue=True)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error cleaning up expired messages: {e}")
                await asyncio.sleep(60)

    async def _update_metrics(self):
        """Background task to update metrics"""
        while self.running:
            try:
                # Update queue size metrics
                for queue_name, queue in self.queues.items():
                    size = await self._get_queue_size(queue_name)
                    self.queue_size.labels(
                        queue_name=queue_name,
                        queue_type=queue.queue_type.value
                    ).set(size)
                
                # Update consumer lag metrics
                for group_id, group in self.consumer_groups.items():
                    for consumer in group.consumers:
                        for topic in consumer.topics:
                            lag = await self._calculate_consumer_lag(topic, group_id)
                            self.consumer_lag.labels(
                                topic=topic,
                                consumer_group=group_id
                            ).set(lag)
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"Error updating metrics: {e}")
                await asyncio.sleep(30)

    async def _rebalance_consumers(self):
        """Background task to rebalance consumer assignments"""
        while self.running:
            try:
                # Simple rebalancing logic - in practice, this would be more sophisticated
                for group_id, group in self.consumer_groups.items():
                    if len(group.consumers) > 0:
                        # Redistribute topics among consumers
                        all_topics = set()
                        for consumer in group.consumers:
                            all_topics.update(consumer.topics)
                        
                        topics_per_consumer = len(all_topics) // len(group.consumers)
                        if topics_per_consumer > 0:
                            logger.debug(f"Rebalanced consumer group {group_id}")
                
                await asyncio.sleep(300)  # Rebalance every 5 minutes
                
            except Exception as e:
                logger.error(f"Error rebalancing consumers: {e}")
                await asyncio.sleep(300)

    async def _process_dead_letter_messages(self):
        """Background task to process dead letter queue messages"""
        while self.running:
            try:
                for dlq_name, messages in self.dead_letter_messages.items():
                    # Log DLQ statistics
                    if messages:
                        logger.info(f"Dead letter queue {dlq_name} has {len(messages)} messages")
                    
                    # Optionally implement DLQ message replay logic here
                    # For now, just maintain the messages for monitoring
                
                await asyncio.sleep(600)  # Check every 10 minutes
                
            except Exception as e:
                logger.error(f"Error processing dead letter messages: {e}")
                await asyncio.sleep(600)

    async def _get_queue_size(self, topic: str) -> int:
        """Get current size of a queue"""
        try:
            queue = self.queues.get(topic)
            if not queue:
                return 0
            
            if queue.queue_type == QueueType.PRIORITY:
                return sum(len(pq) for pq in self.priority_queues.get(topic, []))
            else:
                return len(self.message_store.get(topic, {}))
                
        except Exception as e:
            logger.error(f"Error getting queue size for {topic}: {e}")
            return 0

    async def _calculate_consumer_lag(self, topic: str, group_id: str) -> int:
        """Calculate consumer lag for a topic and group"""
        try:
            # Simplified lag calculation
            queue_size = await self._get_queue_size(topic)
            processing_count = len([m for m in self.processing_messages.values() 
                                  if m.topic == topic])
            
            return max(0, queue_size - processing_count)
            
        except Exception as e:
            logger.error(f"Error calculating consumer lag: {e}")
            return 0

    def _generate_message_id(self, message: Message) -> str:
        """Generate unique message ID"""
        content = f"{message.topic}_{message.timestamp.isoformat()}_{message.payload}"
        return hashlib.md5(content.encode()).hexdigest()

    async def _persist_message(self, message: Message):
        """Persist message to Redis"""
        try:
            if self.redis_client:
                key = f"message:{message.topic}:{message.message_id}"
                serialized_message = pickle.dumps(message)
                
                await self.redis_client.set(key, serialized_message)
                if message.expiry_time:
                    ttl = int((message.expiry_time - datetime.utcnow()).total_seconds())
                    await self.redis_client.expire(key, ttl)
                    
        except Exception as e:
            logger.error(f"Error persisting message: {e}")

    async def _persist_delayed_message(self, message: Message):
        """Persist delayed message to Redis"""
        try:
            if self.redis_client:
                key = f"delayed_message:{message.topic}:{message.message_id}"
                serialized_message = pickle.dumps(message)
                
                await self.redis_client.set(key, serialized_message)
                
                # Set expiry to scheduled time + some buffer
                schedule_time = message.timestamp
                ttl = int((schedule_time - datetime.utcnow()).total_seconds() + 3600)  # +1 hour buffer
                await self.redis_client.expire(key, ttl)
                
        except Exception as e:
            logger.error(f"Error persisting delayed message: {e}")

    async def _remove_persisted_message(self, message: Message):
        """Remove persisted message from Redis"""
        try:
            if self.redis_client:
                key = f"message:{message.topic}:{message.message_id}"
                await self.redis_client.delete(key)
                
        except Exception as e:
            logger.error(f"Error removing persisted message: {e}")

    async def _remove_delayed_message(self, message: Message):
        """Remove delayed message from Redis"""
        try:
            if self.redis_client:
                key = f"delayed_message:{message.topic}:{message.message_id}"
                await self.redis_client.delete(key)
                
        except Exception as e:
            logger.error(f"Error removing delayed message: {e}")

    async def get_broker_health(self) -> Dict[str, Any]:
        """Get health status of the message broker"""
        try:
            health_status = {}
            
            # Queue health
            queues_health = {}
            for queue_name, queue in self.queues.items():
                size = await self._get_queue_size(queue_name)
                queues_health[queue_name] = {
                    "type": queue.queue_type.value,
                    "size": size,
                    "max_size": queue.max_size,
                    "utilization": size / queue.max_size if queue.max_size > 0 else 0,
                    "ttl_seconds": queue.ttl_seconds
                }
            
            # Consumer health
            consumers_health = {}
            for group_id, group in self.consumer_groups.items():
                consumers_health[group_id] = {
                    "active_consumers": len(group.consumers),
                    "topics": list(set().union(*[c.topics for c in group.consumers]))
                }
            
            # Processing statistics
            processing_stats = {
                "active_processing": len(self.processing_messages),
                "delayed_messages": sum(len(msgs) for msgs in self.delayed_messages.values()),
                "dead_letter_messages": sum(len(msgs) for msgs in self.dead_letter_messages.values())
            }
            
            return {
                "status": "healthy",
                "queues": queues_health,
                "consumer_groups": consumers_health,
                "processing": processing_stats,
                "background_tasks": len(self.background_tasks),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def get_broker_metrics(self) -> Dict[str, Any]:
        """Get detailed metrics for the message broker"""
        try:
            metrics = {
                "queues": {},
                "consumer_groups": {},
                "performance": {
                    "total_messages_produced": sum(
                        self.messages_produced._value._value.values()
                    ),
                    "total_messages_consumed": sum(
                        self.messages_consumed._value._value.values()
                    ),
                    "average_processing_time": "Not calculated",  # Would require tracking
                    "active_connections": len(self.processing_messages)
                }
            }
            
            # Queue metrics
            for queue_name, queue in self.queues.items():
                size = await self._get_queue_size(queue_name)
                metrics["queues"][queue_name] = {
                    "size": size,
                    "type": queue.queue_type.value,
                    "utilization_percent": (size / queue.max_size * 100) if queue.max_size > 0 else 0
                }
            
            # Consumer group metrics
            for group_id, group in self.consumer_groups.items():
                total_lag = 0
                for consumer in group.consumers:
                    for topic in consumer.topics:
                        total_lag += await self._calculate_consumer_lag(topic, group_id)
                
                metrics["consumer_groups"][group_id] = {
                    "consumers": len(group.consumers),
                    "total_lag": total_lag
                }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Metrics collection error: {e}")
            return {"error": str(e)}

    async def cleanup(self):
        """Cleanup resources and stop background tasks"""
        try:
            self.running = False
            
            # Cancel background tasks
            for task in self.background_tasks:
                task.cancel()
            
            # Wait for tasks to complete
            if self.background_tasks:
                await asyncio.gather(*self.background_tasks, return_exceptions=True)
            
            # Close Kafka producer
            if self.kafka_producer:
                await self.kafka_producer.stop()
            
            # Close Redis connection
            if self.redis_client:
                await self.redis_client.close()
            
            logger.info("Message Broker cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

# Example usage and testing
async def main():
    broker = EnterpriseMessageBroker()
    await broker.initialize()
    
    # Create test message
    test_message = Message(
        message_id="test_001",
        topic="orders",
        payload={
            "order_id": "ORD-12345",
            "customer_id": "CUST-67890",
            "items": [
                {"product": "Widget A", "quantity": 2, "price": 29.99}
            ],
            "total_amount": 59.98
        },
        headers={
            "source": "web_app",
            "version": "1.0"
        },
        priority=MessagePriority.HIGH,
        correlation_id="corr_001"
    )
    
    # Publish message
    success = await broker.publish_message(test_message)
    print(f"Message published: {success}")
    
    # Register consumer
    async def order_handler(messages):
        for message in messages:
            print(f"Processing order: {message.payload['order_id']}")
            # Simulate processing
            await asyncio.sleep(0.1)
            return True
    
    consumer = Consumer(
        consumer_id="order_processor_001",
        name="Order Processor",
        topics=["orders"],
        group_id="order_processing_group",
        handler=order_handler,
        batch_processing=False,
        max_batch_size=5
    )
    
    await broker.register_consumer(consumer)
    
    # Consume messages
    messages = await broker.consume_messages(consumer.consumer_id)
    if messages:
        print(f"Consumed {len(messages)} messages")
        for message in messages:
            await broker.acknowledge_message(message.message_id, consumer.consumer_id)
    
    # Get health status
    health = await broker.get_broker_health()
    print(f"Broker Health: {json.dumps(health, indent=2)}")
    
    # Wait a bit to see background tasks
    await asyncio.sleep(2)
    
    await broker.cleanup()

if __name__ == "__main__":
    asyncio.run(main())