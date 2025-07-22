"""
Network and communication optimization service for enterprise-scale data quality operations.

This service implements intelligent network optimization, connection management,
request batching, compression, and adaptive protocols for maximum throughput.
"""

import asyncio
import logging
import time
import json
import gzip
import lz4.frame
from typing import Dict, Any, List, Optional, Union, Callable, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import threading
import aiohttp
import websockets
from contextlib import asynccontextmanager
import psutil
import pickle
import base64

from ...domain.entities.quality_profile import DataQualityProfile
from ...domain.value_objects.quality_scores import QualityScores
from ...domain.interfaces.data_quality_interface import DataQualityInterface

logger = logging.getLogger(__name__)


class ProtocolType(Enum):
    """Network protocol types for communication."""
    HTTP = "http"
    HTTPS = "https"
    WEBSOCKET = "websocket"
    GRPC = "grpc"
    TCP = "tcp"
    UDP = "udp"


class CompressionMethod(Enum):
    """Network compression methods."""
    NONE = "none"
    GZIP = "gzip"
    LZ4 = "lz4"
    BROTLI = "brotli"
    ADAPTIVE = "adaptive"


class RequestPriority(Enum):
    """Request priority levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class NetworkEndpoint:
    """Network endpoint configuration and metrics."""
    endpoint_id: str
    url: str
    protocol: ProtocolType
    
    # Connection settings
    max_connections: int = 100
    connection_timeout: float = 30.0
    read_timeout: float = 60.0
    keepalive_timeout: float = 30.0
    
    # Performance metrics
    active_connections: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time_ms: float = 0.0
    
    # Network metrics
    bytes_sent: int = 0
    bytes_received: int = 0
    compression_ratio: float = 1.0
    
    # Health metrics
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    consecutive_failures: int = 0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_requests == 0:
            return 100.0
        return (self.successful_requests / self.total_requests) * 100
    
    @property
    def is_healthy(self) -> bool:
        """Check if endpoint is healthy."""
        return (
            self.consecutive_failures < 5 and
            self.success_rate > 90.0 and
            self.active_connections < self.max_connections
        )
    
    @property
    def utilization(self) -> float:
        """Calculate connection utilization."""
        return (self.active_connections / self.max_connections) * 100


@dataclass
class RequestBatch:
    """Batch of network requests for efficient processing."""
    batch_id: str
    requests: List[Dict[str, Any]] = field(default_factory=list)
    priority: RequestPriority = RequestPriority.MEDIUM
    
    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    max_wait_time_ms: float = 100.0  # Maximum time to wait for more requests
    
    # Batch settings
    max_batch_size: int = 50
    compress_batch: bool = True
    
    @property
    def can_add_request(self) -> bool:
        """Check if batch can accept more requests."""
        return len(self.requests) < self.max_batch_size
    
    @property
    def should_process(self) -> bool:
        """Check if batch should be processed now."""
        if len(self.requests) >= self.max_batch_size:
            return True
        
        age_ms = (datetime.utcnow() - self.created_at).total_seconds() * 1000
        return age_ms >= self.max_wait_time_ms


@dataclass
class NetworkMetrics:
    """Comprehensive network performance metrics."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Throughput metrics
    requests_per_second: float = 0.0
    bytes_per_second: float = 0.0
    concurrent_connections: int = 0
    
    # Latency metrics
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    
    # Error metrics
    error_rate_percent: float = 0.0
    timeout_rate_percent: float = 0.0
    connection_errors: int = 0
    
    # Efficiency metrics
    compression_savings_percent: float = 0.0
    batch_efficiency: float = 0.0
    connection_reuse_rate: float = 0.0
    
    # Resource utilization
    network_cpu_usage_percent: float = 0.0
    network_memory_mb: float = 0.0
    bandwidth_utilization_percent: float = 0.0


class ConnectionPool:
    """Intelligent connection pool with load balancing."""
    
    def __init__(self, endpoint: NetworkEndpoint):
        """Initialize connection pool for endpoint."""
        self.endpoint = endpoint
        self.active_connections: Set[aiohttp.ClientSession] = set()
        self.idle_connections: deque = deque()
        self.connection_queue: asyncio.Queue = asyncio.Queue()
        
        # Pool statistics
        self.connections_created = 0
        self.connections_reused = 0
        self.connections_closed = 0
        
        # Health monitoring
        self.health_check_interval = 30.0  # seconds
        asyncio.create_task(self._health_check_task())
    
    async def get_connection(self) -> aiohttp.ClientSession:
        """Get connection from pool."""
        # Try to reuse idle connection
        if self.idle_connections:
            connection = self.idle_connections.popleft()
            if not connection.closed:
                self.active_connections.add(connection)
                self.connections_reused += 1
                return connection
        
        # Create new connection if under limit
        if len(self.active_connections) < self.endpoint.max_connections:
            connection = await self._create_connection()
            self.active_connections.add(connection)
            self.connections_created += 1
            return connection
        
        # Wait for available connection
        connection = await self.connection_queue.get()
        self.active_connections.add(connection)
        return connection
    
    async def return_connection(self, connection: aiohttp.ClientSession) -> None:
        """Return connection to pool."""
        if connection in self.active_connections:
            self.active_connections.remove(connection)
        
        if not connection.closed:
            self.idle_connections.append(connection)
            
            # Notify waiting requests
            if not self.connection_queue.empty():
                self.connection_queue.put_nowait(connection)
    
    async def _create_connection(self) -> aiohttp.ClientSession:
        """Create new HTTP connection."""
        timeout = aiohttp.ClientTimeout(
            total=self.endpoint.read_timeout,
            connect=self.endpoint.connection_timeout
        )
        
        connector = aiohttp.TCPConnector(
            limit=self.endpoint.max_connections,
            keepalive_timeout=self.endpoint.keepalive_timeout,
            enable_cleanup_closed=True
        )
        
        return aiohttp.ClientSession(
            timeout=timeout,
            connector=connector
        )
    
    async def _health_check_task(self) -> None:
        """Background task for connection health monitoring."""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                
                # Check for stale connections
                stale_connections = []
                for connection in self.idle_connections:
                    if connection.closed:
                        stale_connections.append(connection)
                
                # Remove stale connections
                for connection in stale_connections:
                    self.idle_connections.remove(connection)
                    self.connections_closed += 1
                
                logger.debug(f"Connection pool health check: {len(stale_connections)} stale connections removed")
                
            except Exception as e:
                logger.error(f"Connection pool health check error: {str(e)}")
    
    async def close_all(self) -> None:
        """Close all connections in pool."""
        # Close active connections
        for connection in self.active_connections:
            await connection.close()
        
        # Close idle connections
        while self.idle_connections:
            connection = self.idle_connections.popleft()
            await connection.close()
        
        self.active_connections.clear()


class RequestBatcher:
    """Intelligent request batching for improved throughput."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize request batcher."""
        self.config = config
        self.batches: Dict[str, RequestBatch] = {}
        self.batch_queue: asyncio.Queue = asyncio.Queue()
        
        # Batching configuration
        self.max_batch_size = config.get("max_batch_size", 50)
        self.max_wait_time_ms = config.get("max_wait_time_ms", 100)
        
        # Start batch processing task
        asyncio.create_task(self._batch_processing_task())
    
    async def add_request(self, endpoint_id: str, request_data: Dict[str, Any],
                         priority: RequestPriority = RequestPriority.MEDIUM) -> str:
        """Add request to batch."""
        batch_key = f"{endpoint_id}_{priority.value}"
        
        # Get or create batch
        if batch_key not in self.batches:
            self.batches[batch_key] = RequestBatch(
                batch_id=batch_key,
                priority=priority,
                max_batch_size=self.max_batch_size,
                max_wait_time_ms=self.max_wait_time_ms
            )
        
        batch = self.batches[batch_key]
        batch.requests.append(request_data)
        
        # Check if batch should be processed
        if batch.should_process:
            await self.batch_queue.put(batch)
            del self.batches[batch_key]
        
        return batch.batch_id
    
    async def _batch_processing_task(self) -> None:
        """Background task for processing batches."""
        while True:
            try:
                # Check for ready batches
                ready_batches = []
                for batch_key, batch in list(self.batches.items()):
                    if batch.should_process:
                        ready_batches.append(batch)
                        del self.batches[batch_key]
                
                # Queue ready batches
                for batch in ready_batches:
                    await self.batch_queue.put(batch)
                
                await asyncio.sleep(0.01)  # 10ms check interval
                
            except Exception as e:
                logger.error(f"Batch processing error: {str(e)}")


class AdaptiveCompression:
    """Adaptive compression for network communication."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize adaptive compression."""
        self.config = config
        self.compression_stats: Dict[CompressionMethod, Dict[str, float]] = defaultdict(
            lambda: {"total_time": 0.0, "total_bytes": 0, "compressed_bytes": 0, "count": 0}
        )
    
    def compress_data(self, data: bytes, method: CompressionMethod = CompressionMethod.ADAPTIVE) -> Tuple[bytes, CompressionMethod, float]:
        """Compress data using specified or optimal method."""
        if method == CompressionMethod.ADAPTIVE:
            method = self._choose_optimal_method(data)
        
        if method == CompressionMethod.NONE:
            return data, method, 1.0
        
        start_time = time.time()
        
        try:
            if method == CompressionMethod.GZIP:
                compressed = gzip.compress(data)
            elif method == CompressionMethod.LZ4:
                compressed = lz4.frame.compress(data)
            else:
                compressed = data  # Fallback
                method = CompressionMethod.NONE
            
            compression_time = time.time() - start_time
            compression_ratio = len(compressed) / len(data)
            
            # Update statistics
            stats = self.compression_stats[method]
            stats["total_time"] += compression_time
            stats["total_bytes"] += len(data)
            stats["compressed_bytes"] += len(compressed)
            stats["count"] += 1
            
            return compressed, method, compression_ratio
            
        except Exception as e:
            logger.warning(f"Compression failed: {str(e)}")
            return data, CompressionMethod.NONE, 1.0
    
    def decompress_data(self, data: bytes, method: CompressionMethod) -> bytes:
        """Decompress data using specified method."""
        if method == CompressionMethod.NONE:
            return data
        elif method == CompressionMethod.GZIP:
            return gzip.decompress(data)
        elif method == CompressionMethod.LZ4:
            return lz4.frame.decompress(data)
        else:
            return data
    
    def _choose_optimal_method(self, data: bytes) -> CompressionMethod:
        """Choose optimal compression method based on data characteristics."""
        data_size = len(data)
        
        # Small data: no compression overhead
        if data_size < 1024:
            return CompressionMethod.NONE
        
        # Analyze data type
        try:
            # Try to decode as text
            text = data.decode('utf-8')
            if len(set(text)) / len(text) < 0.3:  # High redundancy
                return CompressionMethod.GZIP  # Good for text
            else:
                return CompressionMethod.LZ4   # Fast for mixed content
        except UnicodeDecodeError:
            # Binary data
            return CompressionMethod.LZ4  # Fast compression for binary


class NetworkCommunicationOptimizer:
    """Comprehensive network communication optimization service."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize network communication optimizer."""
        self.config = config
        
        # Endpoint management
        self.endpoints: Dict[str, NetworkEndpoint] = {}
        self.connection_pools: Dict[str, ConnectionPool] = {}
        
        # Request batching
        self.batcher = RequestBatcher(config)
        
        # Compression
        self.compression = AdaptiveCompression(config)
        
        # Metrics tracking
        self.network_metrics = NetworkMetrics()
        self.metrics_history: List[NetworkMetrics] = []
        
        # Performance tracking
        self.latency_samples: deque = deque(maxlen=1000)
        self.throughput_samples: deque = deque(maxlen=100)
        
        # Configuration
        self.enable_compression = config.get("enable_compression", True)
        self.enable_batching = config.get("enable_batching", True)
        self.enable_keepalive = config.get("enable_keepalive", True)
        self.adaptive_timeouts = config.get("adaptive_timeouts", True)
        
        # Background tasks
        asyncio.create_task(self._network_monitoring_task())
        asyncio.create_task(self._performance_optimization_task())
        asyncio.create_task(self._batch_processor_task())
    
    def register_endpoint(self, endpoint: NetworkEndpoint) -> None:
        """Register network endpoint for optimization."""
        self.endpoints[endpoint.endpoint_id] = endpoint
        self.connection_pools[endpoint.endpoint_id] = ConnectionPool(endpoint)
        
        logger.info(f"Registered endpoint: {endpoint.endpoint_id} ({endpoint.url})")
    
    async def _network_monitoring_task(self) -> None:
        """Background task for network performance monitoring."""
        while True:
            try:
                await asyncio.sleep(60)  # Monitor every minute
                
                # Collect network metrics
                self.network_metrics = await self._collect_network_metrics()
                
                # Record metrics history
                self.metrics_history.append(self.network_metrics)
                
                # Keep only last 24 hours
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                self.metrics_history = [
                    m for m in self.metrics_history if m.timestamp > cutoff_time
                ]
                
                # Analyze performance trends
                await self._analyze_performance_trends()
                
                logger.debug(f"Network monitoring: {self.network_metrics.requests_per_second:.1f} RPS, "
                           f"{self.network_metrics.avg_latency_ms:.1f}ms avg latency")
                
            except Exception as e:
                logger.error(f"Network monitoring error: {str(e)}")
    
    async def _performance_optimization_task(self) -> None:
        """Background task for performance optimization."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Optimize connection pools
                await self._optimize_connection_pools()
                
                # Adjust timeouts based on performance
                if self.adaptive_timeouts:
                    await self._adjust_adaptive_timeouts()
                
                # Optimize compression settings
                await self._optimize_compression_settings()
                
                logger.debug("Performance optimization cycle completed")
                
            except Exception as e:
                logger.error(f"Performance optimization error: {str(e)}")
    
    async def _batch_processor_task(self) -> None:
        """Background task for processing request batches."""
        while True:
            try:
                # Get batch from queue
                batch = await self.batcher.batch_queue.get()
                
                # Process batch
                await self._process_request_batch(batch)
                
            except Exception as e:
                logger.error(f"Batch processing error: {str(e)}")
    
    # Error handling would be managed by interface implementation
    async def send_request(self, endpoint_id: str, method: str, data: Any = None,
                          headers: Dict[str, str] = None, priority: RequestPriority = RequestPriority.MEDIUM) -> Dict[str, Any]:
        """Send optimized network request."""
        endpoint = self.endpoints.get(endpoint_id)
        if not endpoint:
            raise ValueError(f"Endpoint not found: {endpoint_id}")
        
        # Prepare request data
        request_data = {
            "method": method,
            "data": data,
            "headers": headers or {},
            "timestamp": datetime.utcnow().isoformat(),
            "priority": priority.value
        }
        
        # Use batching for non-critical requests
        if self.enable_batching and priority != RequestPriority.CRITICAL:
            batch_id = await self.batcher.add_request(endpoint_id, request_data, priority)
            return {"batch_id": batch_id, "status": "batched"}
        
        # Send individual request
        return await self._send_individual_request(endpoint, request_data)
    
    async def _send_individual_request(self, endpoint: NetworkEndpoint, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send individual network request."""
        start_time = time.time()
        
        try:
            # Get connection from pool
            pool = self.connection_pools[endpoint.endpoint_id]
            connection = await pool.get_connection()
            
            # Prepare request
            method = request_data["method"]
            data = request_data.get("data")
            headers = request_data.get("headers", {})
            
            # Compress data if enabled
            compressed_data = data
            compression_method = CompressionMethod.NONE
            
            if self.enable_compression and data:
                if isinstance(data, (dict, list)):
                    data_bytes = json.dumps(data).encode('utf-8')
                elif isinstance(data, str):
                    data_bytes = data.encode('utf-8')
                else:
                    data_bytes = pickle.dumps(data)
                
                compressed_data, compression_method, compression_ratio = self.compression.compress_data(data_bytes)
                headers["Content-Encoding"] = compression_method.value
                endpoint.compression_ratio = compression_ratio
            
            # Send request
            async with connection.request(method, endpoint.url, data=compressed_data, headers=headers) as response:
                response_data = await response.read()
                
                # Decompress response if needed
                content_encoding = response.headers.get("Content-Encoding")
                if content_encoding and content_encoding in [m.value for m in CompressionMethod]:
                    response_data = self.compression.decompress_data(
                        response_data, CompressionMethod(content_encoding)
                    )
                
                # Parse response
                try:
                    result = json.loads(response_data.decode('utf-8'))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    result = {"raw_data": base64.b64encode(response_data).decode('utf-8')}
                
                # Update metrics
                response_time = (time.time() - start_time) * 1000
                await self._update_endpoint_metrics(endpoint, response_time, True, len(compressed_data or b''), len(response_data))
                
                return {
                    "status": "success",
                    "data": result,
                    "response_time_ms": response_time,
                    "status_code": response.status
                }
            
        except Exception as e:
            # Update failure metrics
            response_time = (time.time() - start_time) * 1000
            await self._update_endpoint_metrics(endpoint, response_time, False)
            
            return {
                "status": "error",
                "error": str(e),
                "response_time_ms": response_time
            }
        
        finally:
            # Return connection to pool
            if 'connection' in locals():
                await pool.return_connection(connection)
    
    async def _process_request_batch(self, batch: RequestBatch) -> None:
        """Process a batch of requests."""
        logger.debug(f"Processing batch {batch.batch_id} with {len(batch.requests)} requests")
        
        # Group requests by endpoint
        endpoint_requests: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        for request in batch.requests:
            endpoint_id = batch.batch_id.split('_')[0]  # Extract endpoint ID from batch ID
            endpoint_requests[endpoint_id].append(request)
        
        # Process each endpoint's requests
        for endpoint_id, requests in endpoint_requests.items():
            endpoint = self.endpoints.get(endpoint_id)
            if not endpoint:
                continue
            
            # Send batch request
            batch_data = {
                "batch_id": batch.batch_id,
                "requests": requests,
                "batch_size": len(requests)
            }
            
            try:
                await self._send_individual_request(endpoint, {
                    "method": "POST",
                    "data": batch_data,
                    "headers": {"Content-Type": "application/json"}
                })
                
                logger.debug(f"Successfully processed batch {batch.batch_id}")
                
            except Exception as e:
                logger.error(f"Failed to process batch {batch.batch_id}: {str(e)}")
    
    async def _update_endpoint_metrics(self, endpoint: NetworkEndpoint, response_time_ms: float,
                                     success: bool, bytes_sent: int = 0, bytes_received: int = 0) -> None:
        """Update endpoint performance metrics."""
        endpoint.total_requests += 1
        
        if success:
            endpoint.successful_requests += 1
            endpoint.last_success = datetime.utcnow()
            endpoint.consecutive_failures = 0
        else:
            endpoint.failed_requests += 1
            endpoint.last_failure = datetime.utcnow()
            endpoint.consecutive_failures += 1
        
        # Update response time
        if endpoint.total_requests == 1:
            endpoint.avg_response_time_ms = response_time_ms
        else:
            endpoint.avg_response_time_ms = (
                (endpoint.avg_response_time_ms * (endpoint.total_requests - 1) + response_time_ms) /
                endpoint.total_requests
            )
        
        # Update network metrics
        endpoint.bytes_sent += bytes_sent
        endpoint.bytes_received += bytes_received
        
        # Store latency sample
        self.latency_samples.append(response_time_ms)
    
    async def _collect_network_metrics(self) -> NetworkMetrics:
        """Collect comprehensive network metrics."""
        # Calculate throughput
        if len(self.throughput_samples) >= 2:
            time_diff = self.throughput_samples[-1] - self.throughput_samples[0]
            requests_per_second = len(self.throughput_samples) / max(1, time_diff)
        else:
            requests_per_second = 0.0
        
        # Calculate latency percentiles
        sorted_latencies = sorted(self.latency_samples) if self.latency_samples else [0.0]
        p95_latency = sorted_latencies[int(len(sorted_latencies) * 0.95)] if sorted_latencies else 0.0
        p99_latency = sorted_latencies[int(len(sorted_latencies) * 0.99)] if sorted_latencies else 0.0
        avg_latency = sum(sorted_latencies) / len(sorted_latencies) if sorted_latencies else 0.0
        
        # Calculate error rates
        total_requests = sum(ep.total_requests for ep in self.endpoints.values())
        total_failures = sum(ep.failed_requests for ep in self.endpoints.values())
        error_rate = (total_failures / total_requests * 100) if total_requests > 0 else 0.0
        
        # Calculate compression savings
        total_compression_ratio = sum(ep.compression_ratio for ep in self.endpoints.values())
        avg_compression_ratio = total_compression_ratio / len(self.endpoints) if self.endpoints else 1.0
        compression_savings = (1.0 - avg_compression_ratio) * 100
        
        return NetworkMetrics(
            requests_per_second=requests_per_second,
            avg_latency_ms=avg_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            error_rate_percent=error_rate,
            compression_savings_percent=compression_savings,
            concurrent_connections=sum(ep.active_connections for ep in self.endpoints.values())
        )
    
    async def _optimize_connection_pools(self) -> None:
        """Optimize connection pool settings based on usage patterns."""
        for endpoint_id, endpoint in self.endpoints.items():
            pool = self.connection_pools[endpoint_id]
            
            # Adjust pool size based on utilization
            utilization = endpoint.utilization
            
            if utilization > 90 and endpoint.max_connections < 500:
                # Increase pool size for high utilization
                endpoint.max_connections = min(500, int(endpoint.max_connections * 1.2))
                logger.info(f"Increased connection pool size for {endpoint_id} to {endpoint.max_connections}")
            
            elif utilization < 30 and endpoint.max_connections > 10:
                # Decrease pool size for low utilization
                endpoint.max_connections = max(10, int(endpoint.max_connections * 0.8))
                logger.info(f"Decreased connection pool size for {endpoint_id} to {endpoint.max_connections}")
    
    async def _adjust_adaptive_timeouts(self) -> None:
        """Adjust timeouts based on performance patterns."""
        if not self.latency_samples:
            return
        
        # Calculate dynamic timeout based on latency patterns
        avg_latency = sum(self.latency_samples) / len(self.latency_samples)
        max_latency = max(self.latency_samples)
        
        # Set timeout to 3x average latency, but at least 5 seconds
        optimal_timeout = max(5.0, (avg_latency * 3) / 1000)
        
        for endpoint in self.endpoints.values():
            if abs(endpoint.read_timeout - optimal_timeout) > 2.0:  # Only update if significant change
                endpoint.read_timeout = optimal_timeout
                logger.debug(f"Adjusted timeout for {endpoint.endpoint_id} to {optimal_timeout:.1f}s")
    
    async def _optimize_compression_settings(self) -> None:
        """Optimize compression settings based on performance data."""
        for method, stats in self.compression.compression_stats.items():
            if stats["count"] > 0:
                avg_ratio = stats["compressed_bytes"] / stats["total_bytes"]
                avg_time_ms = (stats["total_time"] * 1000) / stats["count"]
                
                # If compression isn't providing significant benefit, consider disabling
                if avg_ratio > 0.9 and avg_time_ms > 10:  # Poor compression + slow
                    logger.warning(f"Compression method {method.value} showing poor performance: "
                                 f"{avg_ratio:.2f} ratio, {avg_time_ms:.2f}ms avg time")
    
    # Error handling would be managed by interface implementation
    async def get_network_report(self) -> Dict[str, Any]:
        """Generate comprehensive network performance report."""
        # Endpoint statistics
        endpoint_stats = {}
        for endpoint_id, endpoint in self.endpoints.items():
            pool = self.connection_pools[endpoint_id]
            endpoint_stats[endpoint_id] = {
                "url": endpoint.url,
                "success_rate": endpoint.success_rate,
                "avg_response_time_ms": endpoint.avg_response_time_ms,
                "total_requests": endpoint.total_requests,
                "active_connections": endpoint.active_connections,
                "max_connections": endpoint.max_connections,
                "utilization": endpoint.utilization,
                "is_healthy": endpoint.is_healthy,
                "bytes_sent": endpoint.bytes_sent,
                "bytes_received": endpoint.bytes_received,
                "compression_ratio": endpoint.compression_ratio,
                "pool_stats": {
                    "connections_created": pool.connections_created,
                    "connections_reused": pool.connections_reused,
                    "connections_closed": pool.connections_closed
                }
            }
        
        # Compression statistics
        compression_stats = {}
        for method, stats in self.compression.compression_stats.items():
            if stats["count"] > 0:
                compression_stats[method.value] = {
                    "usage_count": stats["count"],
                    "avg_compression_ratio": stats["compressed_bytes"] / stats["total_bytes"],
                    "avg_time_ms": (stats["total_time"] * 1000) / stats["count"],
                    "total_savings_mb": (stats["total_bytes"] - stats["compressed_bytes"]) / (1024 * 1024)
                }
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "network_performance": {
                "requests_per_second": self.network_metrics.requests_per_second,
                "avg_latency_ms": self.network_metrics.avg_latency_ms,
                "p95_latency_ms": self.network_metrics.p95_latency_ms,
                "p99_latency_ms": self.network_metrics.p99_latency_ms,
                "error_rate_percent": self.network_metrics.error_rate_percent,
                "concurrent_connections": self.network_metrics.concurrent_connections
            },
            "endpoint_statistics": endpoint_stats,
            "compression_statistics": compression_stats,
            "optimization_settings": {
                "compression_enabled": self.enable_compression,
                "batching_enabled": self.enable_batching,
                "keepalive_enabled": self.enable_keepalive,
                "adaptive_timeouts": self.adaptive_timeouts
            },
            "performance_trends": [
                {
                    "timestamp": m.timestamp.isoformat(),
                    "requests_per_second": m.requests_per_second,
                    "avg_latency_ms": m.avg_latency_ms,
                    "error_rate_percent": m.error_rate_percent
                }
                for m in self.metrics_history[-20:]  # Last 20 data points
            ]
        }
    
    async def shutdown(self) -> None:
        """Shutdown the network optimization service."""
        logger.info("Shutting down network communication optimizer...")
        
        # Close all connection pools
        for pool in self.connection_pools.values():
            await pool.close_all()
        
        logger.info("Network communication optimizer shutdown complete")