"""Redis streaming connector for real-time data processing."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import AsyncIterator, Dict, List, Optional, Any
from dataclasses import asdict

from pynomaly.domain.services.streaming_service import StreamRecord, StreamingResult

logger = logging.getLogger(__name__)

# Optional Redis dependencies
try:
    import aioredis
    REDIS_AVAILABLE = True
except ImportError:
    aioredis = None
    REDIS_AVAILABLE = False
    logger.warning("Redis dependencies not available. Install with: pip install aioredis")


class RedisStreamConnector:
    """Redis streams connector for real-time anomaly detection."""
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        input_stream: str = "anomaly_input",
        output_stream: Optional[str] = None,
        consumer_group: str = "pynomaly-group",
        consumer_name: str = "pynomaly-consumer",
        block_time: int = 1000,
        count: int = 10
    ):
        """Initialize Redis streams connector.
        
        Args:
            redis_url: Redis connection URL
            input_stream: Redis stream to consume from
            output_stream: Redis stream to publish to (optional)
            consumer_group: Consumer group name
            consumer_name: Consumer name within group
            block_time: Blocking time for reading streams (ms)
            count: Maximum number of messages to read at once
        """
        if not REDIS_AVAILABLE:
            raise ImportError("Redis dependencies not available. Install with: pip install aioredis")
        
        self.redis_url = redis_url
        self.input_stream = input_stream
        self.output_stream = output_stream
        self.consumer_group = consumer_group
        self.consumer_name = consumer_name
        self.block_time = block_time
        self.count = count
        
        self.redis: Optional[aioredis.Redis] = None
        self._running = False
        self._last_id = ">"  # Start from new messages
        
        self._stats = {
            "messages_consumed": 0,
            "messages_produced": 0,
            "errors": 0,
            "last_message_time": None
        }
    
    async def start(self) -> None:
        """Start Redis connection and setup consumer group."""
        try:
            # Connect to Redis
            self.redis = aioredis.from_url(self.redis_url)
            
            # Test connection
            await self.redis.ping()
            
            # Create consumer group if it doesn't exist
            try:
                await self.redis.xgroup_create(
                    self.input_stream,
                    self.consumer_group,
                    id="0",
                    mkstream=True
                )
                logger.info(f"Created consumer group: {self.consumer_group}")
            except aioredis.ResponseError as e:
                if "BUSYGROUP" in str(e):
                    # Group already exists
                    logger.info(f"Consumer group already exists: {self.consumer_group}")
                else:
                    raise
            
            self._running = True
            logger.info(f"Redis streams connector started for stream: {self.input_stream}")
            
        except Exception as e:
            logger.error(f"Failed to start Redis connector: {e}")
            await self.stop()
            raise
    
    async def stop(self) -> None:
        """Stop Redis connection."""
        self._running = False
        
        try:
            if self.redis:
                await self.redis.close()
            logger.info("Redis connector stopped")
        except Exception as e:
            logger.error(f"Error stopping Redis connector: {e}")
    
    async def consume_stream(self) -> AsyncIterator[StreamRecord]:
        """Consume stream records from Redis stream.
        
        Yields:
            StreamRecord instances from Redis stream messages
        """
        if not self.redis:
            raise RuntimeError("Redis connection not started")
        
        try:
            while self._running:
                try:
                    # Read from consumer group
                    messages = await self.redis.xreadgroup(
                        self.consumer_group,
                        self.consumer_name,
                        {self.input_stream: self._last_id},
                        count=self.count,
                        block=self.block_time
                    )
                    
                    if not messages:
                        continue
                    
                    # Process messages
                    for stream_name, stream_messages in messages:
                        for message_id, fields in stream_messages:
                            try:
                                record = self._message_to_record(message_id, fields)
                                self._stats["messages_consumed"] += 1
                                self._stats["last_message_time"] = datetime.now()
                                
                                # Acknowledge message
                                await self.redis.xack(
                                    self.input_stream,
                                    self.consumer_group,
                                    message_id
                                )
                                
                                yield record
                                
                            except Exception as e:
                                logger.error(f"Failed to process Redis message: {e}")
                                self._stats["errors"] += 1
                                continue
                
                except Exception as e:
                    if self._running:  # Only log if we're supposed to be running
                        logger.error(f"Redis consumption error: {e}")
                        self._stats["errors"] += 1
                        await asyncio.sleep(1)  # Brief pause before retry
        
        except Exception as e:
            logger.error(f"Redis stream consumption failed: {e}")
            raise
    
    async def publish_result(self, result: StreamingResult) -> None:
        """Publish streaming result to Redis stream.
        
        Args:
            result: StreamingResult to publish
        """
        if not self.redis or not self.output_stream:
            return
        
        try:
            # Convert result to Redis fields
            fields = self._result_to_fields(result)
            
            # Add to stream
            message_id = await self.redis.xadd(self.output_stream, fields)
            self._stats["messages_produced"] += 1
            
            logger.debug(f"Published result to Redis stream: {message_id}")
            
        except Exception as e:
            logger.error(f"Failed to publish result to Redis: {e}")
            self._stats["errors"] += 1
    
    async def publish_results_batch(self, results: List[StreamingResult]) -> None:
        """Publish batch of results to Redis stream.
        
        Args:
            results: List of StreamingResults to publish
        """
        if not self.redis or not self.output_stream:
            return
        
        try:
            # Use pipeline for batch operations
            pipe = self.redis.pipeline()
            
            for result in results:
                fields = self._result_to_fields(result)
                pipe.xadd(self.output_stream, fields)
            
            # Execute all commands
            await pipe.execute()
            self._stats["messages_produced"] += len(results)
            
        except Exception as e:
            logger.error(f"Failed to publish batch results to Redis: {e}")
            self._stats["errors"] += 1
    
    def _message_to_record(self, message_id: bytes, fields: Dict[bytes, bytes]) -> StreamRecord:
        """Convert Redis stream message to StreamRecord."""
        # Parse timestamp from message ID
        timestamp_ms = int(message_id.decode().split('-')[0])
        timestamp = datetime.fromtimestamp(timestamp_ms / 1000.0)
        
        # Convert fields to data dictionary
        data = {}
        metadata = {"redis_message_id": message_id.decode()}
        
        for key, value in fields.items():
            key_str = key.decode('utf-8')
            
            try:
                # Try to parse as JSON
                value_str = value.decode('utf-8')
                if value_str.startswith('{') or value_str.startswith('['):
                    parsed_value = json.loads(value_str)
                    data[key_str] = parsed_value
                else:
                    # Try to convert to appropriate type
                    if value_str.lower() in ['true', 'false']:
                        data[key_str] = value_str.lower() == 'true'
                    elif value_str.replace('.', '').replace('-', '').isdigit():
                        data[key_str] = float(value_str) if '.' in value_str else int(value_str)
                    else:
                        data[key_str] = value_str
            except:
                # If all parsing fails, store as string
                data[key_str] = value.decode('utf-8', errors='ignore')
        
        return StreamRecord(
            id=f"redis_{self.input_stream}_{message_id.decode()}",
            timestamp=timestamp,
            data=data,
            metadata=metadata
        )
    
    def _result_to_fields(self, result: StreamingResult) -> Dict[str, str]:
        """Convert StreamingResult to Redis stream fields."""
        fields = {
            "record_id": result.record_id,
            "timestamp": result.timestamp.isoformat(),
            "anomaly_score": str(result.anomaly_score),
            "is_anomaly": str(result.is_anomaly),
            "confidence": str(result.confidence)
        }
        
        # Add explanation if present
        if result.explanation:
            fields["explanation"] = json.dumps(result.explanation)
        
        # Add metadata if present
        if result.metadata:
            fields["metadata"] = json.dumps(result.metadata)
        
        return fields
    
    async def trim_stream(self, stream_name: str, max_length: int = 10000) -> None:
        """Trim Redis stream to maximum length.
        
        Args:
            stream_name: Name of stream to trim
            max_length: Maximum number of messages to keep
        """
        if not self.redis:
            return
        
        try:
            await self.redis.xtrim(stream_name, maxlen=max_length, approximate=True)
        except Exception as e:
            logger.error(f"Failed to trim stream {stream_name}: {e}")
    
    async def get_stream_info(self, stream_name: str) -> Dict[str, Any]:
        """Get information about Redis stream.
        
        Args:
            stream_name: Name of stream
            
        Returns:
            Stream information dictionary
        """
        if not self.redis:
            return {}
        
        try:
            info = await self.redis.xinfo_stream(stream_name)
            return {
                "length": info["length"],
                "first_entry": info["first-entry"],
                "last_entry": info["last-entry"],
                "groups": info["groups"]
            }
        except Exception as e:
            logger.error(f"Failed to get stream info for {stream_name}: {e}")
            return {}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get connector statistics."""
        return self._stats.copy()
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on Redis connection."""
        health = {
            "redis_connected": False,
            "input_stream_exists": False,
            "consumer_group_exists": False,
            "last_error": None
        }
        
        try:
            if self.redis:
                # Test basic connection
                await self.redis.ping()
                health["redis_connected"] = True
                
                # Check if input stream exists
                try:
                    await self.redis.xinfo_stream(self.input_stream)
                    health["input_stream_exists"] = True
                except:
                    pass
                
                # Check if consumer group exists
                try:
                    groups = await self.redis.xinfo_groups(self.input_stream)
                    for group in groups:
                        if group["name"] == self.consumer_group:
                            health["consumer_group_exists"] = True
                            break
                except:
                    pass
        
        except Exception as e:
            health["last_error"] = str(e)
        
        return health