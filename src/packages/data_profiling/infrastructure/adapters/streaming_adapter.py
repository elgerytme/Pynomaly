"""Streaming data adapters for data profiling."""

import pandas as pd
from typing import Dict, Any, Optional, List, Callable, Iterator
from abc import ABC, abstractmethod
import logging
import json
import time
from collections import deque

logger = logging.getLogger(__name__)


class StreamingAdapter(ABC):
    """Abstract base class for streaming data adapters."""
    
    def __init__(self, connection_config: Dict[str, Any]):
        self.connection_config = connection_config
        self._client = None
        self._consumer = None
    
    @abstractmethod
    def connect(self) -> None:
        """Establish connection to streaming service."""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Close connection to streaming service."""
        pass
    
    @abstractmethod
    def consume_messages(self, topic: str, max_messages: int = 1000, 
                        timeout_seconds: int = 30) -> List[Dict[str, Any]]:
        """Consume messages from stream."""
        pass
    
    @abstractmethod
    def get_stream_info(self, topic: str) -> Dict[str, Any]:
        """Get information about the stream."""
        pass
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


class KafkaAdapter(StreamingAdapter):
    """Apache Kafka streaming adapter."""
    
    def connect(self) -> None:
        """Establish Kafka connection."""
        try:
            from kafka import KafkaConsumer, KafkaAdminClient
            from kafka.errors import KafkaError
            
            # Extract connection parameters
            bootstrap_servers = self.connection_config.get('bootstrap_servers', ['localhost:9092'])
            security_protocol = self.connection_config.get('security_protocol', 'PLAINTEXT')
            sasl_mechanism = self.connection_config.get('sasl_mechanism')
            sasl_username = self.connection_config.get('sasl_username')
            sasl_password = self.connection_config.get('sasl_password')
            
            # Build consumer config
            consumer_config = {
                'bootstrap_servers': bootstrap_servers,
                'security_protocol': security_protocol,
                'auto_offset_reset': 'earliest',
                'enable_auto_commit': False,
                'group_id': self.connection_config.get('group_id', 'data_profiling_group'),
                'value_deserializer': lambda x: json.loads(x.decode('utf-8')) if x else None
            }
            
            if sasl_mechanism and sasl_username and sasl_password:
                consumer_config.update({
                    'sasl_mechanism': sasl_mechanism,
                    'sasl_plain_username': sasl_username,
                    'sasl_plain_password': sasl_password
                })
            
            # Create admin client for metadata
            admin_config = {
                'bootstrap_servers': bootstrap_servers,
                'security_protocol': security_protocol
            }
            if sasl_mechanism and sasl_username and sasl_password:
                admin_config.update({
                    'sasl_mechanism': sasl_mechanism,
                    'sasl_plain_username': sasl_username,
                    'sasl_plain_password': sasl_password
                })
            
            self._client = KafkaAdminClient(**admin_config)
            self._consumer_config = consumer_config
            
            # Test connection
            metadata = self._client.describe_configs()
            logger.info("Connected to Kafka successfully")
            
        except ImportError:
            raise ImportError("kafka-python is required for Kafka connectivity")
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            raise
    
    def disconnect(self) -> None:
        """Close Kafka connection."""
        if self._consumer:
            self._consumer.close()
            self._consumer = None
        if self._client:
            self._client.close()
            self._client = None
        logger.info("Disconnected from Kafka")
    
    def consume_messages(self, topic: str, max_messages: int = 1000, 
                        timeout_seconds: int = 30) -> List[Dict[str, Any]]:
        """Consume messages from Kafka topic."""
        if not self._client:
            raise RuntimeError("Not connected to Kafka")
        
        try:
            from kafka import KafkaConsumer
            
            # Create consumer for specific topic
            consumer = KafkaConsumer(topic, **self._consumer_config)
            
            messages = []
            start_time = time.time()
            
            for message in consumer:
                if len(messages) >= max_messages:
                    break
                
                if time.time() - start_time > timeout_seconds:
                    break
                
                message_data = {
                    'topic': message.topic,
                    'partition': message.partition,
                    'offset': message.offset,
                    'timestamp': message.timestamp,
                    'key': message.key.decode('utf-8') if message.key else None,
                    'value': message.value,
                    'headers': dict(message.headers) if message.headers else {}
                }
                messages.append(message_data)
            
            consumer.close()
            logger.info(f"Consumed {len(messages)} messages from Kafka topic {topic}")
            return messages
            
        except Exception as e:
            logger.error(f"Failed to consume Kafka messages: {e}")
            raise
    
    def get_stream_info(self, topic: str) -> Dict[str, Any]:
        """Get information about Kafka topic."""
        if not self._client:
            raise RuntimeError("Not connected to Kafka")
        
        try:
            # Get topic metadata
            metadata = self._client.describe_topics([topic])
            topic_metadata = metadata.get(topic)
            
            if not topic_metadata:
                return {'error': f'Topic {topic} not found'}
            
            return {
                'topic': topic,
                'partitions': len(topic_metadata.partitions),
                'replication_factor': len(topic_metadata.partitions[0].replicas) if topic_metadata.partitions else 0,
                'leader_info': {
                    partition.partition: partition.leader 
                    for partition in topic_metadata.partitions
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get Kafka topic info: {e}")
            return {'error': str(e)}


class KinesisAdapter(StreamingAdapter):
    """AWS Kinesis streaming adapter."""
    
    def connect(self) -> None:
        """Establish Kinesis connection."""
        try:
            import boto3
            from botocore.exceptions import ClientError
            
            # Extract connection parameters
            aws_access_key_id = self.connection_config.get('aws_access_key_id')
            aws_secret_access_key = self.connection_config.get('aws_secret_access_key')
            region_name = self.connection_config.get('region_name', 'us-east-1')
            
            if aws_access_key_id and aws_secret_access_key:
                self._client = boto3.client(
                    'kinesis',
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                    region_name=region_name
                )
            else:
                # Use default credentials
                self._client = boto3.client('kinesis', region_name=region_name)
            
            # Test connection
            self._client.list_streams(Limit=1)
            logger.info("Connected to Kinesis successfully")
            
        except ImportError:
            raise ImportError("boto3 is required for Kinesis connectivity")
        except Exception as e:
            logger.error(f"Failed to connect to Kinesis: {e}")
            raise
    
    def disconnect(self) -> None:
        """Close Kinesis connection."""
        self._client = None
        logger.info("Disconnected from Kinesis")
    
    def consume_messages(self, stream_name: str, max_messages: int = 1000, 
                        timeout_seconds: int = 30) -> List[Dict[str, Any]]:
        """Consume messages from Kinesis stream."""
        if not self._client:
            raise RuntimeError("Not connected to Kinesis")
        
        try:
            # Get stream description
            response = self._client.describe_stream(StreamName=stream_name)
            shards = response['StreamDescription']['Shards']
            
            messages = []
            start_time = time.time()
            
            for shard in shards:
                if len(messages) >= max_messages:
                    break
                
                if time.time() - start_time > timeout_seconds:
                    break
                
                shard_id = shard['ShardId']
                
                # Get shard iterator
                iterator_response = self._client.get_shard_iterator(
                    StreamName=stream_name,
                    ShardId=shard_id,
                    ShardIteratorType='TRIM_HORIZON'
                )
                shard_iterator = iterator_response['ShardIterator']
                
                # Get records
                records_response = self._client.get_records(
                    ShardIterator=shard_iterator,
                    Limit=min(max_messages - len(messages), 10000)  # Kinesis limit
                )
                
                for record in records_response['Records']:
                    if len(messages) >= max_messages:
                        break
                    
                    # Decode data
                    data = record['Data']
                    if isinstance(data, bytes):
                        try:
                            data = json.loads(data.decode('utf-8'))
                        except json.JSONDecodeError:
                            data = data.decode('utf-8')
                    
                    message_data = {
                        'stream_name': stream_name,
                        'shard_id': shard_id,
                        'sequence_number': record['SequenceNumber'],
                        'approximate_arrival_timestamp': record.get('ApproximateArrivalTimestamp'),
                        'partition_key': record['PartitionKey'],
                        'data': data
                    }
                    messages.append(message_data)
            
            logger.info(f"Consumed {len(messages)} messages from Kinesis stream {stream_name}")
            return messages
            
        except Exception as e:
            logger.error(f"Failed to consume Kinesis messages: {e}")
            raise
    
    def get_stream_info(self, stream_name: str) -> Dict[str, Any]:
        """Get information about Kinesis stream."""
        if not self._client:
            raise RuntimeError("Not connected to Kinesis")
        
        try:
            response = self._client.describe_stream(StreamName=stream_name)
            stream_desc = response['StreamDescription']
            
            return {
                'stream_name': stream_name,
                'stream_status': stream_desc['StreamStatus'],
                'shard_count': len(stream_desc['Shards']),
                'retention_period_hours': stream_desc['RetentionPeriodHours'],
                'stream_creation_timestamp': stream_desc['StreamCreationTimestamp'],
                'encryption_type': stream_desc.get('EncryptionType', 'NONE')
            }
            
        except Exception as e:
            logger.error(f"Failed to get Kinesis stream info: {e}")
            return {'error': str(e)}


class StreamingDataProfiler:
    """Profiler for streaming data sources."""
    
    def __init__(self, adapter: StreamingAdapter):
        self.adapter = adapter
        self.message_buffer = deque(maxlen=10000)  # Buffer for incremental profiling
    
    def profile_stream(self, stream_name: str, 
                      sample_size: int = 1000,
                      timeout_seconds: int = 30) -> Dict[str, Any]:
        """Profile a stream by consuming sample messages."""
        try:
            # Get stream info
            stream_info = self.adapter.get_stream_info(stream_name)
            
            # Consume sample messages
            messages = self.adapter.consume_messages(
                stream_name, 
                max_messages=sample_size,
                timeout_seconds=timeout_seconds
            )
            
            if not messages:
                return {
                    'stream_info': stream_info,
                    'sample_size': 0,
                    'profile': None,
                    'success': True,
                    'message': 'No messages available in stream'
                }
            
            # Convert messages to DataFrame for analysis
            df = self._messages_to_dataframe(messages)
            
            # Perform basic profiling
            profile = {
                'row_count': len(df),
                'column_count': len(df.columns),
                'columns': list(df.columns),
                'data_types': df.dtypes.to_dict(),
                'null_counts': df.isnull().sum().to_dict(),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
                'sample_data': df.head().to_dict() if len(df) > 0 else {}
            }
            
            # Add streaming-specific metrics
            profile['message_frequency'] = self._calculate_message_frequency(messages)
            profile['data_freshness'] = self._calculate_data_freshness(messages)
            profile['key_distribution'] = self._analyze_key_distribution(messages)
            
            return {
                'stream_info': stream_info,
                'sample_size': len(messages),
                'profile': profile,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Failed to profile stream {stream_name}: {e}")
            return {
                'stream_name': stream_name,
                'error': str(e),
                'success': False
            }
    
    def _messages_to_dataframe(self, messages: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert stream messages to DataFrame."""
        if not messages:
            return pd.DataFrame()
        
        # Extract data from messages
        records = []
        for msg in messages:
            if isinstance(msg.get('value'), dict):
                # Kafka-style message with JSON value
                record = msg['value'].copy()
                record['_message_timestamp'] = msg.get('timestamp')
                record['_message_key'] = msg.get('key')
            elif isinstance(msg.get('data'), dict):
                # Kinesis-style message with JSON data
                record = msg['data'].copy()
                record['_message_timestamp'] = msg.get('approximate_arrival_timestamp')
                record['_partition_key'] = msg.get('partition_key')
            else:
                # Raw message
                record = {
                    'message_data': str(msg.get('value', msg.get('data', ''))),
                    '_message_timestamp': msg.get('timestamp', msg.get('approximate_arrival_timestamp')),
                    '_message_key': msg.get('key', msg.get('partition_key'))
                }
            
            records.append(record)
        
        return pd.DataFrame(records)
    
    def _calculate_message_frequency(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate message frequency statistics."""
        if len(messages) < 2:
            return {'messages_per_second': 0, 'interval_seconds': 0}
        
        # Extract timestamps
        timestamps = []
        for msg in messages:
            ts = msg.get('timestamp', msg.get('approximate_arrival_timestamp'))
            if ts:
                if isinstance(ts, (int, float)):
                    timestamps.append(ts / 1000 if ts > 1e10 else ts)  # Convert from ms if needed
                else:
                    timestamps.append(time.mktime(ts.timetuple()) if hasattr(ts, 'timetuple') else 0)
        
        if len(timestamps) < 2:
            return {'messages_per_second': 0, 'interval_seconds': 0}
        
        timestamps.sort()
        interval_seconds = timestamps[-1] - timestamps[0]
        
        if interval_seconds > 0:
            messages_per_second = len(messages) / interval_seconds
        else:
            messages_per_second = 0
        
        return {
            'messages_per_second': round(messages_per_second, 2),
            'interval_seconds': round(interval_seconds, 2),
            'total_messages': len(messages)
        }
    
    def _calculate_data_freshness(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate data freshness metrics."""
        current_time = time.time()
        freshness_scores = []
        
        for msg in messages:
            ts = msg.get('timestamp', msg.get('approximate_arrival_timestamp'))
            if ts:
                if isinstance(ts, (int, float)):
                    msg_time = ts / 1000 if ts > 1e10 else ts
                else:
                    msg_time = time.mktime(ts.timetuple()) if hasattr(ts, 'timetuple') else current_time
                
                age_seconds = current_time - msg_time
                freshness_scores.append(age_seconds)
        
        if not freshness_scores:
            return {'average_age_seconds': 0, 'oldest_message_age_seconds': 0}
        
        return {
            'average_age_seconds': round(sum(freshness_scores) / len(freshness_scores), 2),
            'oldest_message_age_seconds': round(max(freshness_scores), 2),
            'newest_message_age_seconds': round(min(freshness_scores), 2)
        }
    
    def _analyze_key_distribution(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze distribution of message keys/partition keys."""
        keys = []
        for msg in messages:
            key = msg.get('key', msg.get('partition_key'))
            if key:
                keys.append(str(key))
        
        if not keys:
            return {'unique_keys': 0, 'total_keys': 0, 'key_distribution': {}}
        
        from collections import Counter
        key_counts = Counter(keys)
        
        return {
            'unique_keys': len(key_counts),
            'total_keys': len(keys),
            'key_distribution': dict(key_counts.most_common(10)),  # Top 10 keys
            'keys_with_single_message': sum(1 for count in key_counts.values() if count == 1)
        }
    
    def profile_continuous(self, stream_name: str, 
                          callback: Callable[[Dict[str, Any]], None],
                          duration_seconds: int = 60,
                          batch_size: int = 100) -> None:
        """Profile stream continuously for a specified duration."""
        start_time = time.time()
        total_messages = 0
        
        logger.info(f"Starting continuous profiling of {stream_name} for {duration_seconds} seconds")
        
        while time.time() - start_time < duration_seconds:
            try:
                # Consume batch of messages
                messages = self.adapter.consume_messages(
                    stream_name,
                    max_messages=batch_size,
                    timeout_seconds=5
                )
                
                if messages:
                    total_messages += len(messages)
                    
                    # Add to buffer
                    self.message_buffer.extend(messages)
                    
                    # Create incremental profile
                    profile_data = {
                        'batch_size': len(messages),
                        'total_messages': total_messages,
                        'elapsed_time': time.time() - start_time,
                        'messages_per_second': total_messages / (time.time() - start_time),
                        'buffer_size': len(self.message_buffer)
                    }
                    
                    # Convert latest messages to DataFrame for analysis
                    if len(messages) > 0:
                        df = self._messages_to_dataframe(messages)
                        profile_data['latest_batch'] = {
                            'columns': list(df.columns),
                            'data_types': df.dtypes.to_dict(),
                            'null_counts': df.isnull().sum().to_dict()
                        }
                    
                    # Call user callback with profile data
                    callback(profile_data)
                
                time.sleep(1)  # Brief pause between batches
                
            except Exception as e:
                logger.error(f"Error during continuous profiling: {e}")
                break
        
        logger.info(f"Continuous profiling completed. Total messages processed: {total_messages}")


def get_streaming_adapter(provider: str, connection_config: Dict[str, Any]) -> StreamingAdapter:
    """Factory function to get the appropriate streaming adapter."""
    adapters = {
        'kafka': KafkaAdapter,
        'kinesis': KinesisAdapter,
        'aws_kinesis': KinesisAdapter
    }
    
    provider_lower = provider.lower()
    if provider_lower not in adapters:
        raise ValueError(f"Unsupported streaming provider: {provider}. Supported: {list(adapters.keys())}")
    
    return adapters[provider_lower](connection_config)