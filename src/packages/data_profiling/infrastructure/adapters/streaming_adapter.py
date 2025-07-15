import pandas as pd
import json
from typing import Dict, Any, Optional, List, Callable, Iterator
import logging
from abc import ABC, abstractmethod
from queue import Queue
import threading
import time
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class StreamingAdapter(ABC):
    """Abstract base class for streaming data adapters."""
    
    def __init__(self, connection_config: Dict[str, Any], **kwargs):
        self.connection_config = connection_config
        self.kwargs = kwargs
        self.is_connected = False
        self.is_consuming = False
        self.message_queue = Queue()
        self.consumer_thread = None
        self.stop_event = threading.Event()
    
    @abstractmethod
    def connect(self) -> None:
        """Establish connection to streaming source."""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Close connection to streaming source."""
        pass
    
    @abstractmethod
    def consume_messages(self) -> Iterator[Dict[str, Any]]:
        """Consume messages from streaming source."""
        pass
    
    def start_consuming(self, callback: Optional[Callable] = None) -> None:
        """Start consuming messages in a separate thread."""
        if self.is_consuming:
            logger.warning("Already consuming messages")
            return
        
        self.is_consuming = True
        self.stop_event.clear()
        
        def consume_loop():
            try:
                for message in self.consume_messages():
                    if self.stop_event.is_set():
                        break
                    
                    self.message_queue.put(message)
                    
                    if callback:
                        callback(message)
                    
                    if self.message_queue.qsize() > 10000:  # Prevent memory overflow
                        logger.warning("Message queue getting full, consider processing faster")
                        
            except Exception as e:
                logger.error(f"Error in consume loop: {e}")
            finally:
                self.is_consuming = False
        
        self.consumer_thread = threading.Thread(target=consume_loop)
        self.consumer_thread.start()
    
    def stop_consuming(self) -> None:
        """Stop consuming messages."""
        if not self.is_consuming:
            return
        
        self.stop_event.set()
        
        if self.consumer_thread:
            self.consumer_thread.join(timeout=5)
            if self.consumer_thread.is_alive():
                logger.warning("Consumer thread did not stop gracefully")
        
        self.is_consuming = False
    
    def get_messages(self, max_messages: int = 100, timeout: float = 1.0) -> List[Dict[str, Any]]:
        """Get messages from the queue."""
        messages = []
        start_time = time.time()
        
        while len(messages) < max_messages and time.time() - start_time < timeout:
            try:
                message = self.message_queue.get(timeout=0.1)
                messages.append(message)
                self.message_queue.task_done()
            except:
                break
        
        return messages
    
    def get_dataframe(self, max_messages: int = 1000, timeout: float = 5.0) -> pd.DataFrame:
        """Get messages as a pandas DataFrame."""
        messages = self.get_messages(max_messages, timeout)
        
        if not messages:
            return pd.DataFrame()
        
        # Convert messages to DataFrame
        try:
            df = pd.DataFrame(messages)
            return df
        except Exception as e:
            logger.error(f"Failed to convert messages to DataFrame: {e}")
            return pd.DataFrame()


class KafkaAdapter(StreamingAdapter):
    """Apache Kafka streaming adapter."""
    
    def __init__(self, connection_config: Dict[str, Any], topic: str, **kwargs):
        super().__init__(connection_config, **kwargs)
        self.topic = topic
        self.consumer = None
        self.producer = None
    
    def connect(self) -> None:
        """Establish connection to Kafka."""
        try:
            from kafka import KafkaConsumer, KafkaProducer
            
            # Create consumer
            consumer_config = {
                'bootstrap_servers': self.connection_config['bootstrap_servers'],
                'auto_offset_reset': self.connection_config.get('auto_offset_reset', 'earliest'),
                'enable_auto_commit': self.connection_config.get('enable_auto_commit', True),
                'group_id': self.connection_config.get('group_id', 'pynomaly-profiler'),
                'value_deserializer': lambda m: json.loads(m.decode('utf-8')) if m else None
            }
            
            # Add security configurations if provided
            if 'security_protocol' in self.connection_config:
                consumer_config['security_protocol'] = self.connection_config['security_protocol']
            if 'sasl_mechanism' in self.connection_config:
                consumer_config['sasl_mechanism'] = self.connection_config['sasl_mechanism']
            if 'sasl_plain_username' in self.connection_config:
                consumer_config['sasl_plain_username'] = self.connection_config['sasl_plain_username']
            if 'sasl_plain_password' in self.connection_config:
                consumer_config['sasl_plain_password'] = self.connection_config['sasl_plain_password']
            
            self.consumer = KafkaConsumer(self.topic, **consumer_config)
            
            # Create producer (optional)
            producer_config = {
                'bootstrap_servers': self.connection_config['bootstrap_servers'],
                'value_serializer': lambda v: json.dumps(v).encode('utf-8')
            }
            
            self.producer = KafkaProducer(**producer_config)
            
            self.is_connected = True
            logger.info(f"Connected to Kafka topic: {self.topic}")
            
        except ImportError:
            raise ImportError("kafka-python is required for Kafka connectivity")
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            raise
    
    def disconnect(self) -> None:
        """Close connection to Kafka."""
        if self.consumer:
            self.consumer.close()
            self.consumer = None
        
        if self.producer:
            self.producer.close()
            self.producer = None
        
        self.is_connected = False
        logger.info("Disconnected from Kafka")
    
    def consume_messages(self) -> Iterator[Dict[str, Any]]:
        """Consume messages from Kafka topic."""
        if not self.consumer:
            raise RuntimeError("Not connected to Kafka")
        
        try:
            for message in self.consumer:
                if self.stop_event.is_set():
                    break
                
                # Parse message
                parsed_message = {
                    'timestamp': datetime.fromtimestamp(message.timestamp / 1000),
                    'topic': message.topic,
                    'partition': message.partition,
                    'offset': message.offset,
                    'key': message.key.decode('utf-8') if message.key else None,
                    'value': message.value,
                    'headers': dict(message.headers) if message.headers else {}
                }
                
                yield parsed_message
                
        except Exception as e:
            logger.error(f"Error consuming Kafka messages: {e}")
            raise
    
    def get_topic_metadata(self) -> Dict[str, Any]:
        """Get metadata about the Kafka topic."""
        if not self.consumer:
            raise RuntimeError("Not connected to Kafka")
        
        try:
            metadata = self.consumer.list_consumer_group_offsets()
            partitions = self.consumer.partitions_for_topic(self.topic)
            
            return {
                'topic': self.topic,
                'partitions': list(partitions) if partitions else [],
                'partition_count': len(partitions) if partitions else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get Kafka topic metadata: {e}")
            return {}


class KinesisAdapter(StreamingAdapter):
    """Amazon Kinesis streaming adapter."""
    
    def __init__(self, connection_config: Dict[str, Any], stream_name: str, **kwargs):
        super().__init__(connection_config, **kwargs)
        self.stream_name = stream_name
        self.kinesis_client = None
        self.shard_iterator = None
    
    def connect(self) -> None:
        """Establish connection to Kinesis."""
        try:
            import boto3
            
            # Create Kinesis client
            self.kinesis_client = boto3.client(
                'kinesis',
                aws_access_key_id=self.connection_config.get('aws_access_key_id'),
                aws_secret_access_key=self.connection_config.get('aws_secret_access_key'),
                aws_session_token=self.connection_config.get('aws_session_token'),
                region_name=self.connection_config.get('region', 'us-east-1')
            )
            
            # Get stream description
            stream_description = self.kinesis_client.describe_stream(StreamName=self.stream_name)
            logger.info(f"Connected to Kinesis stream: {self.stream_name}")
            
            # Get shard iterator for the first shard
            shards = stream_description['StreamDescription']['Shards']
            if shards:
                shard_id = shards[0]['ShardId']
                shard_iterator_response = self.kinesis_client.get_shard_iterator(
                    StreamName=self.stream_name,
                    ShardId=shard_id,
                    ShardIteratorType='TRIM_HORIZON'
                )
                self.shard_iterator = shard_iterator_response['ShardIterator']
            
            self.is_connected = True
            
        except ImportError:
            raise ImportError("boto3 is required for Kinesis connectivity")
        except Exception as e:
            logger.error(f"Failed to connect to Kinesis: {e}")
            raise
    
    def disconnect(self) -> None:
        """Close connection to Kinesis."""
        self.kinesis_client = None
        self.shard_iterator = None
        self.is_connected = False
        logger.info("Disconnected from Kinesis")
    
    def consume_messages(self) -> Iterator[Dict[str, Any]]:
        """Consume messages from Kinesis stream."""
        if not self.kinesis_client or not self.shard_iterator:
            raise RuntimeError("Not connected to Kinesis")
        
        try:
            current_shard_iterator = self.shard_iterator
            
            while not self.stop_event.is_set():
                response = self.kinesis_client.get_records(
                    ShardIterator=current_shard_iterator,
                    Limit=100
                )
                
                records = response.get('Records', [])
                current_shard_iterator = response.get('NextShardIterator')
                
                if not records:
                    time.sleep(1)  # Wait if no records
                    continue
                
                for record in records:
                    # Parse record
                    parsed_message = {
                        'timestamp': record['ApproximateArrivalTimestamp'],
                        'sequence_number': record['SequenceNumber'],
                        'partition_key': record['PartitionKey'],
                        'data': json.loads(record['Data'].decode('utf-8'))
                    }
                    
                    yield parsed_message
                
                if not current_shard_iterator:
                    break
                    
        except Exception as e:
            logger.error(f"Error consuming Kinesis messages: {e}")
            raise
    
    def get_stream_metadata(self) -> Dict[str, Any]:
        """Get metadata about the Kinesis stream."""
        if not self.kinesis_client:
            raise RuntimeError("Not connected to Kinesis")
        
        try:
            response = self.kinesis_client.describe_stream(StreamName=self.stream_name)
            stream_description = response['StreamDescription']
            
            return {
                'stream_name': self.stream_name,
                'stream_status': stream_description['StreamStatus'],
                'shard_count': len(stream_description['Shards']),
                'retention_period': stream_description['RetentionPeriodHours']
            }
            
        except Exception as e:
            logger.error(f"Failed to get Kinesis stream metadata: {e}")
            return {}


class EventHubAdapter(StreamingAdapter):
    """Azure Event Hub streaming adapter."""
    
    def __init__(self, connection_config: Dict[str, Any], event_hub_name: str, **kwargs):
        super().__init__(connection_config, **kwargs)
        self.event_hub_name = event_hub_name
        self.consumer_client = None
    
    def connect(self) -> None:
        """Establish connection to Event Hub."""
        try:
            from azure.eventhub import EventHubConsumerClient
            
            # Create consumer client
            self.consumer_client = EventHubConsumerClient.from_connection_string(
                conn_str=self.connection_config['connection_string'],
                consumer_group=self.connection_config.get('consumer_group', '$Default'),
                eventhub_name=self.event_hub_name
            )
            
            self.is_connected = True
            logger.info(f"Connected to Event Hub: {self.event_hub_name}")
            
        except ImportError:
            raise ImportError("azure-eventhub is required for Event Hub connectivity")
        except Exception as e:
            logger.error(f"Failed to connect to Event Hub: {e}")
            raise
    
    def disconnect(self) -> None:
        """Close connection to Event Hub."""
        if self.consumer_client:
            self.consumer_client.close()
            self.consumer_client = None
        
        self.is_connected = False
        logger.info("Disconnected from Event Hub")
    
    def consume_messages(self) -> Iterator[Dict[str, Any]]:
        """Consume messages from Event Hub."""
        if not self.consumer_client:
            raise RuntimeError("Not connected to Event Hub")
        
        try:
            def on_event(partition_context, event):
                # Parse event
                parsed_message = {
                    'timestamp': event.enqueued_time,
                    'partition_id': partition_context.partition_id,
                    'offset': event.offset,
                    'sequence_number': event.sequence_number,
                    'body': json.loads(event.body_as_str()),
                    'properties': event.properties
                }
                
                self.message_queue.put(parsed_message)
                partition_context.update_checkpoint(event)
            
            # Start receiving
            with self.consumer_client:
                self.consumer_client.receive(
                    on_event=on_event,
                    starting_position="-1"  # Start from beginning
                )
                
        except Exception as e:
            logger.error(f"Error consuming Event Hub messages: {e}")
            raise
    
    def get_event_hub_metadata(self) -> Dict[str, Any]:
        """Get metadata about the Event Hub."""
        if not self.consumer_client:
            raise RuntimeError("Not connected to Event Hub")
        
        try:
            # Get partition information
            partition_ids = self.consumer_client.get_partition_ids()
            
            return {
                'event_hub_name': self.event_hub_name,
                'partition_count': len(partition_ids),
                'partition_ids': list(partition_ids)
            }
            
        except Exception as e:
            logger.error(f"Failed to get Event Hub metadata: {e}")
            return {}


class StreamingProfiler:
    """Helper class for profiling streaming data sources."""
    
    def __init__(self, adapter: StreamingAdapter):
        self.adapter = adapter
        self.profile_data = []
        self.profile_lock = threading.Lock()
    
    def start_profiling(self, duration_seconds: int = 60, 
                       max_messages: int = 10000) -> None:
        """Start profiling streaming data."""
        if not self.adapter.is_connected:
            self.adapter.connect()
        
        # Start consuming messages
        self.adapter.start_consuming(callback=self._profile_message)
        
        # Wait for specified duration
        start_time = time.time()
        while (time.time() - start_time < duration_seconds and 
               len(self.profile_data) < max_messages):
            time.sleep(1)
        
        # Stop consuming
        self.adapter.stop_consuming()
    
    def _profile_message(self, message: Dict[str, Any]) -> None:
        """Profile a single message."""
        with self.profile_lock:
            self.profile_data.append(message)
    
    def get_profile_dataframe(self) -> pd.DataFrame:
        """Get profiled data as DataFrame."""
        with self.profile_lock:
            if not self.profile_data:
                return pd.DataFrame()
            
            return pd.DataFrame(self.profile_data)
    
    def get_streaming_statistics(self) -> Dict[str, Any]:
        """Get statistics about the streaming data."""
        with self.profile_lock:
            if not self.profile_data:
                return {}
            
            df = pd.DataFrame(self.profile_data)
            
            # Calculate statistics
            stats = {
                'total_messages': len(self.profile_data),
                'unique_keys': df['key'].nunique() if 'key' in df.columns else 0,
                'message_rate': len(self.profile_data) / 60,  # messages per minute
                'data_size_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
                'column_count': len(df.columns),
                'null_percentages': df.isnull().mean().to_dict()
            }
            
            # Add timestamp analysis if available
            if 'timestamp' in df.columns:
                timestamps = pd.to_datetime(df['timestamp'])
                stats['time_span'] = (timestamps.max() - timestamps.min()).total_seconds()
                stats['message_frequency'] = len(self.profile_data) / stats['time_span']
            
            return stats
    
    def clear_profile_data(self) -> None:
        """Clear accumulated profile data."""
        with self.profile_lock:
            self.profile_data.clear()


def get_streaming_adapter(provider: str, connection_config: Dict[str, Any], 
                         stream_name: str, **kwargs) -> StreamingAdapter:
    """Factory function to get the appropriate streaming adapter."""
    providers = {
        'kafka': KafkaAdapter,
        'kinesis': KinesisAdapter,
        'aws_kinesis': KinesisAdapter,
        'eventhub': EventHubAdapter,
        'azure_eventhub': EventHubAdapter
    }
    
    provider_lower = provider.lower()
    if provider_lower not in providers:
        raise ValueError(f"Unsupported streaming provider: {provider}. Supported: {list(providers.keys())}")
    
    return providers[provider_lower](connection_config, stream_name, **kwargs)