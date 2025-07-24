"""
Real-Time Inference Engine

High-performance real-time inference system with streaming capabilities,
dynamic model routing, and comprehensive monitoring.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
import uuid
from collections import defaultdict, deque
import time

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
import structlog
from concurrent.futures import ThreadPoolExecutor
import aioredis
from prometheus_client import Counter, Histogram, Gauge

from mlops.domain.entities.model import Model, ModelVersion
from mlops.infrastructure.feature_store.feature_store import FeatureStore


class InferenceMode(Enum):
    """Inference execution modes."""
    SYNCHRONOUS = "synchronous"
    ASYNCHRONOUS = "asynchronous"
    STREAMING = "streaming"
    BATCH = "batch"


class RoutingStrategy(Enum):
    """Model routing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED = "weighted"
    RANDOM = "random"
    PERFORMANCE_BASED = "performance_based"


@dataclass
class InferenceRequest:
    """Real-time inference request."""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_id: str = ""
    model_version: Optional[str] = None
    features: Dict[str, Any] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    priority: int = 0  # Higher priority = processed first
    timeout_ms: int = 5000
    require_explanation: bool = False
    callback_url: Optional[str] = None


@dataclass
class InferenceResponse:
    """Real-time inference response."""
    request_id: str
    model_id: str
    model_version: str
    predictions: Union[List[float], List[int], List[str]]
    probabilities: Optional[List[List[float]]] = None
    confidence_scores: Optional[List[float]] = None
    explanations: Optional[List[Dict[str, Any]]] = None
    feature_importance: Optional[Dict[str, float]] = None
    processing_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    model_performance_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class StreamingBatch:
    """Batch of streaming inference requests."""
    batch_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    requests: List[InferenceRequest] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    batch_size: int = 0
    
    def __post_init__(self):
        self.batch_size = len(self.requests)


@dataclass
class ModelEndpoint:
    """Model serving endpoint configuration."""
    model_id: str
    model_version: str
    endpoint_url: str
    weight: float = 1.0
    max_concurrent_requests: int = 100
    current_connections: int = 0
    health_status: str = "healthy"
    last_health_check: datetime = field(default_factory=datetime.utcnow)
    avg_response_time_ms: float = 0.0
    success_rate: float = 1.0
    total_requests: int = 0


class RealTimeInferenceEngine:
    """High-performance real-time inference engine."""
    
    def __init__(self, 
                 feature_store: FeatureStore,
                 redis_url: str = "redis://localhost:6379",
                 max_batch_size: int = 32,
                 batch_timeout_ms: int = 100,
                 max_workers: int = 8):
        
        self.feature_store = feature_store
        self.redis_url = redis_url
        self.max_batch_size = max_batch_size
        self.batch_timeout_ms = batch_timeout_ms
        self.max_workers = max_workers
        
        self.logger = structlog.get_logger(__name__)
        
        # Model registry and routing
        self.model_endpoints: Dict[str, List[ModelEndpoint]] = defaultdict(list)
        self.routing_strategy = RoutingStrategy.ROUND_ROBIN
        self.routing_counters: Dict[str, int] = defaultdict(int)
        
        # Request processing
        self.request_queue: asyncio.Queue = asyncio.Queue()
        self.streaming_batches: Dict[str, StreamingBatch] = {}
        self.active_requests: Dict[str, InferenceRequest] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Caching
        self.redis_client: Optional[aioredis.Redis] = None
        self.cache_ttl_seconds = 300  # 5 minutes
        
        # Performance monitoring
        self.metrics = self._init_metrics()
        self.performance_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "p99_latency_ms": 0.0,
            "requests_per_second": 0.0
        }
        
        # Request latency tracking for percentiles
        self.latency_window = deque(maxlen=10000)
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        self.is_running = False
        
        # Feature preprocessing
        self.feature_preprocessors: Dict[str, Callable] = {}
        
        # A/B testing support
        self.ab_test_configs: Dict[str, Dict] = {}
        
    def _init_metrics(self) -> Dict[str, Any]:
        """Initialize Prometheus metrics."""
        return {
            "requests_total": Counter(
                'realtime_inference_requests_total',
                'Total inference requests',
                ['model_id', 'model_version', 'status', 'mode']
            ),
            "request_duration": Histogram(
                'realtime_inference_request_duration_seconds',
                'Request processing duration',
                ['model_id', 'model_version', 'mode']
            ),
            "batch_size": Histogram(
                'realtime_inference_batch_size',
                'Batch size for streaming inference',
                ['model_id']
            ),
            "queue_size": Gauge(
                'realtime_inference_queue_size',
                'Current queue size'
            ),
            "active_connections": Gauge(
                'realtime_inference_active_connections',
                'Active connections per endpoint',
                ['model_id', 'endpoint_url']
            ),
            "cache_hits": Counter(
                'realtime_inference_cache_hits_total',
                'Cache hits',
                ['model_id']
            ),
            "cache_misses": Counter(
                'realtime_inference_cache_misses_total',
                'Cache misses',
                ['model_id']
            )
        }
    
    async def start(self) -> None:
        """Start the inference engine."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Initialize Redis connection
        try:
            self.redis_client = aioredis.from_url(self.redis_url)
            await self.redis_client.ping()
            self.logger.info("Redis connection established")
        except Exception as e:
            self.logger.warning("Redis connection failed, caching disabled", error=str(e))
            self.redis_client = None
        
        # Start background tasks
        self.background_tasks = [
            asyncio.create_task(self._request_processor()),
            asyncio.create_task(self._batch_processor()),
            asyncio.create_task(self._health_checker()),
            asyncio.create_task(self._metrics_collector()),
            asyncio.create_task(self._cleanup_expired_requests())
        ]
        
        self.logger.info("Real-time inference engine started")
    
    async def stop(self) -> None:
        """Stop the inference engine."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        self.logger.info("Real-time inference engine stopped")
    
    async def register_model_endpoint(self, 
                                    model_id: str,
                                    model_version: str,
                                    endpoint_url: str,
                                    weight: float = 1.0,
                                    max_concurrent: int = 100) -> None:
        """Register a model endpoint for inference."""
        
        endpoint = ModelEndpoint(
            model_id=model_id,
            model_version=model_version,
            endpoint_url=endpoint_url,
            weight=weight,
            max_concurrent_requests=max_concurrent
        )
        
        self.model_endpoints[model_id].append(endpoint)
        
        self.logger.info(
            "Model endpoint registered",
            model_id=model_id,
            model_version=model_version,
            endpoint_url=endpoint_url,
            weight=weight
        )
    
    async def predict(self, 
                     request: InferenceRequest,
                     mode: InferenceMode = InferenceMode.SYNCHRONOUS) -> InferenceResponse:
        """Make a real-time prediction."""
        
        start_time = time.time()
        request.timestamp = datetime.utcnow()
        
        try:
            # Add to active requests
            self.active_requests[request.request_id] = request
            
            # Update metrics
            self.metrics["queue_size"].set(len(self.active_requests))
            
            if mode == InferenceMode.SYNCHRONOUS:
                response = await self._process_sync_request(request)
            elif mode == InferenceMode.ASYNCHRONOUS:
                response = await self._process_async_request(request)
            elif mode == InferenceMode.STREAMING:
                response = await self._process_streaming_request(request)
            else:
                raise ValueError(f"Unsupported inference mode: {mode}")
            
            # Record successful request
            processing_time = (time.time() - start_time) * 1000
            response.processing_time_ms = processing_time
            
            self._update_performance_stats(processing_time, success=True)
            
            self.metrics["requests_total"].labels(
                model_id=request.model_id,
                model_version=response.model_version,
                status="success",
                mode=mode.value
            ).inc()
            
            self.metrics["request_duration"].labels(
                model_id=request.model_id,
                model_version=response.model_version,
                mode=mode.value
            ).observe(processing_time / 1000)
            
            return response
            
        except Exception as e:
            # Record failed request
            processing_time = (time.time() - start_time) * 1000
            self._update_performance_stats(processing_time, success=False)
            
            self.metrics["requests_total"].labels(
                model_id=request.model_id,
                model_version="unknown",
                status="error",
                mode=mode.value
            ).inc()
            
            self.logger.error(
                "Inference request failed",
                request_id=request.request_id,
                model_id=request.model_id,
                error=str(e)
            )
            
            # Return error response
            return InferenceResponse(
                request_id=request.request_id,
                model_id=request.model_id,
                model_version="unknown",
                predictions=[],
                processing_time_ms=processing_time,
                warnings=[f"Inference failed: {str(e)}"]
            )
        
        finally:
            # Remove from active requests
            if request.request_id in self.active_requests:
                del self.active_requests[request.request_id]
            
            self.metrics["queue_size"].set(len(self.active_requests))
    
    async def _process_sync_request(self, request: InferenceRequest) -> InferenceResponse:
        """Process synchronous inference request."""
        
        # Check cache first
        cache_key = self._get_cache_key(request)
        cached_response = await self._get_cached_response(cache_key, request.model_id)
        
        if cached_response:
            self.metrics["cache_hits"].labels(model_id=request.model_id).inc()
            return cached_response
        
        self.metrics["cache_misses"].labels(model_id=request.model_id).inc()
        
        # Get model endpoint
        endpoint = await self._select_model_endpoint(request.model_id)
        if not endpoint:
            raise ValueError(f"No healthy endpoints available for model {request.model_id}")
        
        # Prepare features
        features = await self._prepare_features(request)
        
        # Make inference
        response = await self._make_model_inference(endpoint, features, request)
        
        # Cache response
        await self._cache_response(cache_key, response)
        
        return response
    
    async def _process_async_request(self, request: InferenceRequest) -> InferenceResponse:
        """Process asynchronous inference request."""
        
        # Add to queue for background processing
        await self.request_queue.put(request)
        
        # Return immediate response with request ID
        return InferenceResponse(
            request_id=request.request_id,
            model_id=request.model_id,
            model_version="pending",
            predictions=[],
            processing_time_ms=0.0,
            metadata={"status": "queued", "estimated_completion": "30s"}
        )
    
    async def _process_streaming_request(self, request: InferenceRequest) -> InferenceResponse:
        """Process streaming inference request."""
        
        # Add to current batch or create new batch
        batch_id = await self._add_to_streaming_batch(request)
        
        return InferenceResponse(
            request_id=request.request_id,
            model_id=request.model_id,
            model_version="batched",
            predictions=[],
            processing_time_ms=0.0,
            metadata={"status": "batched", "batch_id": batch_id}
        )
    
    async def _prepare_features(self, request: InferenceRequest) -> Dict[str, Any]:
        """Prepare features for inference."""
        
        features = request.features.copy()
        
        # Apply feature preprocessing if configured
        if request.model_id in self.feature_preprocessors:
            preprocessor = self.feature_preprocessors[request.model_id]
            features = await asyncio.get_event_loop().run_in_executor(
                self.executor, preprocessor, features
            )
        
        # Fetch additional features from feature store if needed
        if request.feature_names and hasattr(request, 'entity_id'):
            entity_id = request.metadata.get('entity_id')
            if entity_id:
                try:
                    feature_df = await self.feature_store.get_features(
                        request.feature_names, [entity_id]
                    )
                    if not feature_df.empty:
                        feature_row = feature_df.iloc[0].to_dict()
                        features.update(feature_row)
                except Exception as e:
                    self.logger.warning(
                        "Failed to fetch features from feature store",
                        entity_id=entity_id,
                        error=str(e)
                    )
        
        return features
    
    async def _make_model_inference(self,
                                  endpoint: ModelEndpoint,
                                  features: Dict[str, Any],
                                  request: InferenceRequest) -> InferenceResponse:
        """Make inference using model endpoint."""
        
        # Increment connection count
        endpoint.current_connections += 1
        self.metrics["active_connections"].labels(
            model_id=endpoint.model_id,
            endpoint_url=endpoint.endpoint_url
        ).set(endpoint.current_connections)
        
        try:
            start_time = time.time()
            
            # In a real implementation, this would make HTTP call to the model endpoint
            # For now, simulate inference
            await asyncio.sleep(0.01)  # Simulate processing time
            
            # Mock prediction result
            predictions = [0.8, 0.2]  # Binary classification example
            probabilities = [[0.8, 0.2]]
            confidence_scores = [0.8]
            
            processing_time = (time.time() - start_time) * 1000
            
            # Update endpoint stats
            endpoint.total_requests += 1
            endpoint.avg_response_time_ms = (
                (endpoint.avg_response_time_ms * (endpoint.total_requests - 1) + processing_time) /
                endpoint.total_requests
            )
            
            response = InferenceResponse(
                request_id=request.request_id,
                model_id=endpoint.model_id,
                model_version=endpoint.model_version,
                predictions=predictions,
                probabilities=probabilities,
                confidence_scores=confidence_scores,
                processing_time_ms=processing_time,
                model_performance_metrics={
                    "endpoint_avg_latency_ms": endpoint.avg_response_time_ms,
                    "endpoint_success_rate": endpoint.success_rate
                }
            )
            
            # Add explanations if requested
            if request.require_explanation:
                response.explanations = await self._generate_explanations(
                    features, predictions, endpoint
                )
            
            return response
            
        except Exception as e:
            # Update endpoint health
            endpoint.success_rate = max(0.0, endpoint.success_rate - 0.01)
            raise e
        
        finally:
            # Decrement connection count
            endpoint.current_connections -= 1
            self.metrics["active_connections"].labels(
                model_id=endpoint.model_id,
                endpoint_url=endpoint.endpoint_url
            ).set(endpoint.current_connections)
    
    async def _select_model_endpoint(self, model_id: str) -> Optional[ModelEndpoint]:
        """Select the best endpoint for the model."""
        
        endpoints = self.model_endpoints.get(model_id, [])
        if not endpoints:
            return None
        
        # Filter healthy endpoints
        healthy_endpoints = [
            ep for ep in endpoints 
            if (ep.health_status == "healthy" and 
                ep.current_connections < ep.max_concurrent_requests)
        ]
        
        if not healthy_endpoints:
            return None
        
        # Apply routing strategy
        if self.routing_strategy == RoutingStrategy.ROUND_ROBIN:
            idx = self.routing_counters[model_id] % len(healthy_endpoints)
            self.routing_counters[model_id] += 1
            return healthy_endpoints[idx]
        
        elif self.routing_strategy == RoutingStrategy.LEAST_CONNECTIONS:
            return min(healthy_endpoints, key=lambda ep: ep.current_connections)
        
        elif self.routing_strategy == RoutingStrategy.PERFORMANCE_BASED:
            # Select based on response time and success rate
            scores = []
            for ep in healthy_endpoints:
                # Lower latency and higher success rate = better score
                score = ep.success_rate / (ep.avg_response_time_ms + 1)
                scores.append((score, ep))
            
            return max(scores, key=lambda x: x[0])[1]
        
        elif self.routing_strategy == RoutingStrategy.WEIGHTED:
            # Weighted random selection
            import random
            weights = [ep.weight for ep in healthy_endpoints]
            return random.choices(healthy_endpoints, weights=weights)[0]
        
        else:  # RANDOM
            import random
            return random.choice(healthy_endpoints)
    
    async def _generate_explanations(self,
                                   features: Dict[str, Any],
                                   predictions: List[float],
                                   endpoint: ModelEndpoint) -> List[Dict[str, Any]]:
        """Generate model explanations."""
        
        # Mock explanation generation
        # In production, this would use SHAP, LIME, or other explainability tools
        explanations = []
        
        # Feature importance (mock)
        feature_importance = {}
        for feature_name, value in features.items():
            if isinstance(value, (int, float)):
                # Mock importance score
                importance = abs(value) * 0.1
                feature_importance[feature_name] = importance
        
        explanations.append({
            "prediction_index": 0,
            "prediction_value": predictions[0] if predictions else 0.0,
            "feature_importance": feature_importance,
            "explanation_method": "mock_shap",
            "confidence": 0.8
        })
        
        return explanations
    
    async def _add_to_streaming_batch(self, request: InferenceRequest) -> str:
        """Add request to streaming batch."""
        
        # Find existing batch or create new one
        model_batches = [
            batch for batch in self.streaming_batches.values()
            if (len([r for r in batch.requests if r.model_id == request.model_id]) < self.max_batch_size and
                (datetime.utcnow() - batch.created_at).total_seconds() * 1000 < self.batch_timeout_ms)
        ]
        
        if model_batches:
            # Add to existing batch
            batch = model_batches[0]
            batch.requests.append(request)
            batch.batch_size = len(batch.requests)
        else:
            # Create new batch
            batch = StreamingBatch(requests=[request])
            self.streaming_batches[batch.batch_id] = batch
        
        return batch.batch_id
    
    async def _request_processor(self) -> None:
        """Background task to process queued requests."""
        while self.is_running:
            try:
                # Get request from queue with timeout
                request = await asyncio.wait_for(
                    self.request_queue.get(), timeout=1.0
                )
                
                # Process request
                response = await self._process_sync_request(request)
                
                # If callback URL provided, send response
                if request.callback_url:
                    await self._send_callback(request.callback_url, response)
                
                # Store result for later retrieval
                if self.redis_client:
                    await self.redis_client.setex(
                        f"async_result:{request.request_id}",
                        3600,  # 1 hour TTL
                        json.dumps(response.__dict__, default=str)
                    )
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error("Error in request processor", error=str(e))
    
    async def _batch_processor(self) -> None:
        """Background task to process streaming batches."""
        while self.is_running:
            try:
                await asyncio.sleep(self.batch_timeout_ms / 1000)
                
                # Process ready batches
                ready_batches = []
                current_time = datetime.utcnow()
                
                for batch_id, batch in list(self.streaming_batches.items()):
                    batch_age_ms = (current_time - batch.created_at).total_seconds() * 1000
                    
                    if (batch.batch_size >= self.max_batch_size or 
                        batch_age_ms >= self.batch_timeout_ms):
                        ready_batches.append(batch)
                        del self.streaming_batches[batch_id]
                
                # Process batches
                for batch in ready_batches:
                    await self._process_batch(batch)
                
            except Exception as e:
                self.logger.error("Error in batch processor", error=str(e))
    
    async def _process_batch(self, batch: StreamingBatch) -> None:
        """Process a batch of streaming requests."""
        
        try:
            # Group requests by model
            model_requests = defaultdict(list)
            for request in batch.requests:
                model_requests[request.model_id].append(request)
            
            # Process each model group
            for model_id, requests in model_requests.items():
                endpoint = await self._select_model_endpoint(model_id)
                if not endpoint:
                    self.logger.warning(f"No endpoint available for model {model_id}")
                    continue
                
                # Prepare batch features
                batch_features = []
                for request in requests:
                    features = await self._prepare_features(request)
                    batch_features.append(features)
                
                # Make batch inference
                batch_predictions = await self._make_batch_inference(
                    endpoint, batch_features, requests
                )
                
                # Send individual responses
                for i, (request, prediction) in enumerate(zip(requests, batch_predictions)):
                    if request.callback_url:
                        await self._send_callback(request.callback_url, prediction)
            
            self.metrics["batch_size"].labels(
                model_id=list(model_requests.keys())[0] if model_requests else "unknown"
            ).observe(batch.batch_size)
            
        except Exception as e:
            self.logger.error(
                "Batch processing failed",
                batch_id=batch.batch_id,
                batch_size=batch.batch_size,
                error=str(e)
            )
    
    async def _make_batch_inference(self,
                                  endpoint: ModelEndpoint,
                                  batch_features: List[Dict[str, Any]],
                                  requests: List[InferenceRequest]) -> List[InferenceResponse]:
        """Make batch inference."""
        
        start_time = time.time()
        
        # Simulate batch processing
        await asyncio.sleep(0.05)  # Simulate batch processing time
        
        # Generate responses
        responses = []
        for i, request in enumerate(requests):
            processing_time = (time.time() - start_time) * 1000
            
            response = InferenceResponse(
                request_id=request.request_id,
                model_id=endpoint.model_id,
                model_version=endpoint.model_version,
                predictions=[0.7, 0.3],  # Mock predictions
                probabilities=[[0.7, 0.3]],
                confidence_scores=[0.7],
                processing_time_ms=processing_time,
                metadata={"batch_processed": True}
            )
            responses.append(response)
        
        return responses
    
    async def _health_checker(self) -> None:
        """Background task to check endpoint health."""
        while self.is_running:
            try:
                for model_id, endpoints in self.model_endpoints.items():
                    for endpoint in endpoints:
                        # Simple health check (in production, would ping actual endpoint)
                        if endpoint.success_rate > 0.8:
                            endpoint.health_status = "healthy"
                        elif endpoint.success_rate > 0.5:
                            endpoint.health_status = "degraded"
                        else:
                            endpoint.health_status = "unhealthy"
                        
                        endpoint.last_health_check = datetime.utcnow()
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error("Error in health checker", error=str(e))
    
    async def _metrics_collector(self) -> None:
        """Background task to collect and update performance metrics."""
        while self.is_running:
            try:
                # Calculate percentiles from latency window
                if self.latency_window:
                    latencies = list(self.latency_window)
                    self.performance_stats["p95_latency_ms"] = np.percentile(latencies, 95)
                    self.performance_stats["p99_latency_ms"] = np.percentile(latencies, 99)
                
                await asyncio.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                self.logger.error("Error in metrics collector", error=str(e))
    
    async def _cleanup_expired_requests(self) -> None:
        """Background task to clean up expired requests."""
        while self.is_running:
            try:
                current_time = datetime.utcnow()
                expired_requests = []
                
                for request_id, request in self.active_requests.items():
                    age_ms = (current_time - request.timestamp).total_seconds() * 1000
                    if age_ms > request.timeout_ms:
                        expired_requests.append(request_id)
                
                for request_id in expired_requests:
                    del self.active_requests[request_id]
                    self.logger.warning(
                        "Request expired and removed",
                        request_id=request_id
                    )
                
                await asyncio.sleep(60)  # Clean up every minute
                
            except Exception as e:
                self.logger.error("Error in cleanup task", error=str(e))
    
    def _get_cache_key(self, request: InferenceRequest) -> str:
        """Generate cache key for request."""
        # Create hash of features and model info
        import hashlib
        
        key_data = {
            "model_id": request.model_id,
            "features": request.features,
            "feature_names": sorted(request.feature_names)
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return f"inference_cache:{hashlib.sha256(key_str.encode()).hexdigest()}"
    
    async def _get_cached_response(self, cache_key: str, model_id: str) -> Optional[InferenceResponse]:
        """Get cached response."""
        if not self.redis_client:
            return None
        
        try:
            cached_data = await self.redis_client.get(cache_key)
            if cached_data:
                response_data = json.loads(cached_data)
                return InferenceResponse(**response_data)
        except Exception as e:
            self.logger.warning("Cache retrieval failed", error=str(e))
        
        return None
    
    async def _cache_response(self, cache_key: str, response: InferenceResponse) -> None:
        """Cache response."""
        if not self.redis_client:
            return
        
        try:
            response_data = response.__dict__.copy()
            # Convert datetime objects to strings
            response_data["timestamp"] = response.timestamp.isoformat()
            
            await self.redis_client.setex(
                cache_key,
                self.cache_ttl_seconds,
                json.dumps(response_data, default=str)
            )
        except Exception as e:
            self.logger.warning("Cache storage failed", error=str(e))
    
    async def _send_callback(self, callback_url: str, response: InferenceResponse) -> None:
        """Send callback with inference response."""
        try:
            # In production, would make HTTP POST to callback URL
            self.logger.info(
                "Callback sent",
                callback_url=callback_url,
                request_id=response.request_id
            )
        except Exception as e:
            self.logger.error(
                "Callback failed",
                callback_url=callback_url,
                error=str(e)
            )
    
    def _update_performance_stats(self, processing_time_ms: float, success: bool) -> None:
        """Update performance statistics."""
        self.performance_stats["total_requests"] += 1
        
        if success:
            self.performance_stats["successful_requests"] += 1
        else:
            self.performance_stats["failed_requests"] += 1
        
        # Update average latency
        total_requests = self.performance_stats["total_requests"]
        current_avg = self.performance_stats["avg_latency_ms"]
        self.performance_stats["avg_latency_ms"] = (
            (current_avg * (total_requests - 1) + processing_time_ms) / total_requests
        )
        
        # Add to latency window for percentile calculation
        self.latency_window.append(processing_time_ms)
    
    def register_feature_preprocessor(self, model_id: str, preprocessor: Callable) -> None:
        """Register feature preprocessor for a model."""
        self.feature_preprocessors[model_id] = preprocessor
        
        self.logger.info(
            "Feature preprocessor registered",
            model_id=model_id
        )
    
    def set_routing_strategy(self, strategy: RoutingStrategy) -> None:
        """Set model routing strategy."""
        self.routing_strategy = strategy
        
        self.logger.info(
            "Routing strategy updated",
            strategy=strategy.value
        )
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        return self.performance_stats.copy()
    
    async def get_endpoint_status(self) -> Dict[str, Any]:
        """Get status of all model endpoints."""
        status = {}
        
        for model_id, endpoints in self.model_endpoints.items():
            status[model_id] = []
            for endpoint in endpoints:
                status[model_id].append({
                    "endpoint_url": endpoint.endpoint_url,
                    "model_version": endpoint.model_version,
                    "health_status": endpoint.health_status,
                    "current_connections": endpoint.current_connections,
                    "max_connections": endpoint.max_concurrent_requests,
                    "avg_response_time_ms": endpoint.avg_response_time_ms,
                    "success_rate": endpoint.success_rate,
                    "total_requests": endpoint.total_requests,
                    "last_health_check": endpoint.last_health_check.isoformat()
                })
        
        return status
    
    async def predict_streaming(self, 
                              model_id: str,
                              feature_stream: AsyncGenerator[Dict[str, Any], None]) -> AsyncGenerator[InferenceResponse, None]:
        """Process streaming predictions."""
        
        async for features in feature_stream:
            request = InferenceRequest(
                model_id=model_id,
                features=features,
                timestamp=datetime.utcnow()
            )
            
            response = await self.predict(request, InferenceMode.STREAMING)
            yield response