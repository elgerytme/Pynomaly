"""Use case for streaming anomaly detection with backpressure handling.

This module provides production-ready streaming anomaly detection capabilities with
sophisticated backpressure management, adaptive buffering, and real-time analytics.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Union
from uuid import uuid4

import numpy as np
import pandas as pd

from pynomaly.domain.entities import Dataset, Detector, DetectionResult
from pynomaly.domain.exceptions import DomainError, ValidationError
from pynomaly.domain.value_objects import AnomalyScore

logger = logging.getLogger(__name__)


class StreamingStrategy(Enum):
    """Strategies for streaming anomaly detection."""
    
    REAL_TIME = "real_time"          # Process each sample immediately
    MICRO_BATCH = "micro_batch"      # Process small batches frequently
    ADAPTIVE_BATCH = "adaptive_batch" # Dynamic batch sizing based on load
    WINDOWED = "windowed"            # Sliding window processing
    ENSEMBLE_STREAM = "ensemble_stream" # Ensemble streaming with multiple detectors


class BackpressureStrategy(Enum):
    """Strategies for handling backpressure in streaming systems."""
    
    DROP_OLDEST = "drop_oldest"      # Drop oldest samples when buffer full
    DROP_NEWEST = "drop_newest"      # Drop newest samples when buffer full
    ADAPTIVE_SAMPLING = "adaptive_sampling" # Reduce sampling rate under pressure
    CIRCUIT_BREAKER = "circuit_breaker" # Temporarily halt processing
    ELASTIC_SCALING = "elastic_scaling" # Scale processing resources


class StreamingMode(Enum):
    """Modes for streaming detection operation."""
    
    CONTINUOUS = "continuous"        # Continuous processing
    BURST = "burst"                 # Handle burst traffic
    SCHEDULED = "scheduled"         # Scheduled batch processing
    EVENT_DRIVEN = "event_driven"   # Process on specific events


@dataclass
class StreamingSample:
    """Individual sample in streaming pipeline."""
    
    id: str = field(default_factory=lambda: str(uuid4()))
    data: Union[np.ndarray, Dict[str, Any]] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # Higher priority = processed first


@dataclass
class StreamingResult:
    """Result from streaming detection."""
    
    sample_id: str
    prediction: int  # 0 = normal, 1 = anomaly
    anomaly_score: float
    confidence: float
    processing_time: float
    detector_id: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamingConfiguration:
    """Configuration for streaming detection."""
    
    strategy: StreamingStrategy = StreamingStrategy.ADAPTIVE_BATCH
    backpressure_strategy: BackpressureStrategy = BackpressureStrategy.ADAPTIVE_SAMPLING
    mode: StreamingMode = StreamingMode.CONTINUOUS
    
    # Buffer configuration
    max_buffer_size: int = 10000
    min_batch_size: int = 1
    max_batch_size: int = 100
    batch_timeout_ms: int = 100
    
    # Backpressure thresholds
    high_watermark: float = 0.8  # Start backpressure at 80% buffer full
    low_watermark: float = 0.3   # Resume normal processing at 30% buffer full
    
    # Performance settings
    max_processing_time_ms: int = 1000
    max_concurrent_batches: int = 5
    adaptive_scaling_enabled: bool = True
    
    # Quality settings
    enable_quality_monitoring: bool = True
    quality_check_interval_ms: int = 5000
    drift_detection_enabled: bool = True
    
    # Output configuration
    enable_result_buffering: bool = True
    result_buffer_size: int = 1000
    enable_metrics_collection: bool = True


@dataclass
class StreamingRequest:
    """Request for streaming detection setup."""
    
    detector_id: str
    configuration: StreamingConfiguration
    callback_handlers: Dict[str, Callable] = field(default_factory=dict)
    enable_ensemble: bool = False
    ensemble_detector_ids: List[str] = field(default_factory=list)


@dataclass
class StreamingResponse:
    """Response from streaming detection setup."""
    
    success: bool
    stream_id: str = ""
    configuration: Optional[StreamingConfiguration] = None
    error_message: Optional[str] = None
    performance_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamingMetrics:
    """Real-time metrics for streaming detection."""
    
    stream_id: str
    samples_processed: int = 0
    samples_dropped: int = 0
    anomalies_detected: int = 0
    average_processing_time: float = 0.0
    current_buffer_size: int = 0
    buffer_utilization: float = 0.0
    throughput_per_second: float = 0.0
    backpressure_active: bool = False
    circuit_breaker_open: bool = False
    last_updated: float = field(default_factory=time.time)
    error_rate: float = 0.0
    quality_score: float = 1.0


class StreamingDetectionUseCase:
    """Use case for streaming anomaly detection with backpressure handling."""
    
    def __init__(
        self,
        detector_repository,
        adapter_registry,
        ensemble_service=None,
        enable_distributed_processing: bool = False,
        max_concurrent_streams: int = 10
    ):
        """Initialize streaming detection use case.
        
        Args:
            detector_repository: Repository for detector management
            adapter_registry: Registry for algorithm adapters
            ensemble_service: Optional ensemble service for multi-detector streaming
            enable_distributed_processing: Enable distributed processing capabilities
            max_concurrent_streams: Maximum number of concurrent streams
        """
        self.detector_repository = detector_repository
        self.adapter_registry = adapter_registry
        self.ensemble_service = ensemble_service
        self.enable_distributed_processing = enable_distributed_processing
        self.max_concurrent_streams = max_concurrent_streams
        
        # Active streams management
        self._active_streams: Dict[str, Dict[str, Any]] = {}
        self._stream_metrics: Dict[str, StreamingMetrics] = {}
        self._stream_tasks: Dict[str, asyncio.Task] = {}
        
        # Global configuration
        self._global_config = {
            "enable_monitoring": True,
            "metrics_collection_interval": 1.0,
            "health_check_interval": 5.0,
            "auto_cleanup_inactive": True,
            "cleanup_interval": 30.0
        }
        
        logger.info("Streaming detection use case initialized")
    
    async def start_streaming(
        self, request: StreamingRequest
    ) -> StreamingResponse:
        """Start streaming anomaly detection.
        
        Args:
            request: Streaming detection request
            
        Returns:
            Streaming response with stream ID and configuration
        """
        try:
            logger.info(f"Starting streaming detection for detector {request.detector_id}")
            
            # Validate request
            validation_result = await self._validate_streaming_request(request)
            if not validation_result["valid"]:
                return StreamingResponse(
                    success=False,
                    error_message=validation_result["error"]
                )
            
            # Check concurrent stream limits
            if len(self._active_streams) >= self.max_concurrent_streams:
                return StreamingResponse(
                    success=False,
                    error_message=f"Maximum concurrent streams ({self.max_concurrent_streams}) reached"
                )
            
            # Generate stream ID
            stream_id = f"stream_{uuid4()}"
            
            # Initialize stream state
            stream_state = {
                "request": request,
                "buffer": deque(maxlen=request.configuration.max_buffer_size),
                "result_buffer": deque(maxlen=request.configuration.result_buffer_size),
                "backpressure_active": False,
                "circuit_breaker_open": False,
                "processing_stats": {
                    "samples_processed": 0,
                    "processing_times": deque(maxlen=1000),
                    "start_time": time.time()
                },
                "quality_monitor": {
                    "last_check": time.time(),
                    "quality_score": 1.0,
                    "drift_detected": False
                }
            }
            
            self._active_streams[stream_id] = stream_state
            
            # Initialize metrics
            self._stream_metrics[stream_id] = StreamingMetrics(stream_id=stream_id)
            
            # Start processing task
            task = asyncio.create_task(
                self._process_stream(stream_id, request.configuration)
            )
            self._stream_tasks[stream_id] = task
            
            # Start monitoring if enabled
            if request.configuration.enable_metrics_collection:
                asyncio.create_task(self._monitor_stream(stream_id))
            
            logger.info(f"Streaming detection started with ID: {stream_id}")
            
            return StreamingResponse(
                success=True,
                stream_id=stream_id,
                configuration=request.configuration,
                performance_metrics=self._get_initial_metrics()
            )
            
        except Exception as e:
            logger.error(f"Error starting streaming detection: {str(e)}")
            return StreamingResponse(
                success=False,
                error_message=f"Failed to start streaming: {str(e)}"
            )
    
    async def stop_streaming(self, stream_id: str) -> bool:
        """Stop streaming detection.
        
        Args:
            stream_id: ID of stream to stop
            
        Returns:
            True if successfully stopped, False otherwise
        """
        try:
            if stream_id not in self._active_streams:
                logger.warning(f"Stream {stream_id} not found")
                return False
            
            # Cancel processing task
            if stream_id in self._stream_tasks:
                task = self._stream_tasks[stream_id]
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                del self._stream_tasks[stream_id]
            
            # Cleanup stream state
            del self._active_streams[stream_id]
            
            # Keep metrics for a while for analysis
            if stream_id in self._stream_metrics:
                self._stream_metrics[stream_id].last_updated = time.time()
            
            logger.info(f"Streaming detection stopped for ID: {stream_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping streaming detection: {str(e)}")
            return False
    
    async def add_sample(
        self, stream_id: str, sample: StreamingSample
    ) -> bool:
        """Add sample to streaming pipeline.
        
        Args:
            stream_id: Stream ID to add sample to
            sample: Sample to process
            
        Returns:
            True if sample was added, False if dropped due to backpressure
        """
        try:
            if stream_id not in self._active_streams:
                logger.warning(f"Stream {stream_id} not found")
                return False
            
            stream_state = self._active_streams[stream_id]
            buffer = stream_state["buffer"]
            config = stream_state["request"].configuration
            
            # Check buffer capacity and apply backpressure strategy
            current_utilization = len(buffer) / config.max_buffer_size
            
            if current_utilization >= config.high_watermark:
                # Apply backpressure strategy
                if config.backpressure_strategy == BackpressureStrategy.DROP_OLDEST:
                    if buffer:
                        buffer.popleft()  # Remove oldest
                    buffer.append(sample)
                    self._stream_metrics[stream_id].samples_dropped += 1
                    
                elif config.backpressure_strategy == BackpressureStrategy.DROP_NEWEST:
                    # Don't add the new sample
                    self._stream_metrics[stream_id].samples_dropped += 1
                    return False
                    
                elif config.backpressure_strategy == BackpressureStrategy.ADAPTIVE_SAMPLING:
                    # Implement adaptive sampling (simplified)
                    if np.random.random() > 0.5:  # Drop 50% of samples under pressure
                        self._stream_metrics[stream_id].samples_dropped += 1
                        return False
                    buffer.append(sample)
                    
                elif config.backpressure_strategy == BackpressureStrategy.CIRCUIT_BREAKER:
                    # Open circuit breaker
                    stream_state["circuit_breaker_open"] = True
                    self._stream_metrics[stream_id].circuit_breaker_open = True
                    self._stream_metrics[stream_id].samples_dropped += 1
                    return False
                    
                else:
                    buffer.append(sample)
                
                stream_state["backpressure_active"] = True
                self._stream_metrics[stream_id].backpressure_active = True
            else:
                # Normal operation
                buffer.append(sample)
                
                # Reset backpressure if utilization is low enough
                if current_utilization <= config.low_watermark:
                    stream_state["backpressure_active"] = False
                    stream_state["circuit_breaker_open"] = False
                    self._stream_metrics[stream_id].backpressure_active = False
                    self._stream_metrics[stream_id].circuit_breaker_open = False
            
            # Update metrics
            self._stream_metrics[stream_id].current_buffer_size = len(buffer)
            self._stream_metrics[stream_id].buffer_utilization = current_utilization
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding sample to stream {stream_id}: {str(e)}")
            return False
    
    async def get_results(
        self, stream_id: str, max_results: int = 100
    ) -> List[StreamingResult]:
        """Get results from streaming detection.
        
        Args:
            stream_id: Stream ID to get results from
            max_results: Maximum number of results to return
            
        Returns:
            List of streaming results
        """
        try:
            if stream_id not in self._active_streams:
                return []
            
            stream_state = self._active_streams[stream_id]
            result_buffer = stream_state["result_buffer"]
            
            results = []
            for _ in range(min(max_results, len(result_buffer))):
                if result_buffer:
                    results.append(result_buffer.popleft())
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting results from stream {stream_id}: {str(e)}")
            return []
    
    async def get_stream_metrics(self, stream_id: str) -> Optional[StreamingMetrics]:
        """Get real-time metrics for a stream.
        
        Args:
            stream_id: Stream ID to get metrics for
            
        Returns:
            Streaming metrics or None if stream not found
        """
        return self._stream_metrics.get(stream_id)
    
    async def list_active_streams(self) -> List[str]:
        """List all active stream IDs.
        
        Returns:
            List of active stream IDs
        """
        return list(self._active_streams.keys())
    
    # Private helper methods
    
    async def _validate_streaming_request(
        self, request: StreamingRequest
    ) -> Dict[str, Any]:
        """Validate streaming request."""
        try:
            # Check detector exists
            detector = await self.detector_repository.get(request.detector_id)
            if not detector:
                return {"valid": False, "error": f"Detector {request.detector_id} not found"}
            
            if not detector.is_fitted:
                return {"valid": False, "error": f"Detector {request.detector_id} is not fitted"}
            
            # Validate ensemble configuration
            if request.enable_ensemble:
                if not request.ensemble_detector_ids:
                    return {"valid": False, "error": "Ensemble enabled but no ensemble detector IDs provided"}
                
                for ensemble_detector_id in request.ensemble_detector_ids:
                    ensemble_detector = await self.detector_repository.get(ensemble_detector_id)
                    if not ensemble_detector:
                        return {"valid": False, "error": f"Ensemble detector {ensemble_detector_id} not found"}
                    if not ensemble_detector.is_fitted:
                        return {"valid": False, "error": f"Ensemble detector {ensemble_detector_id} is not fitted"}
            
            # Validate configuration
            config = request.configuration
            if config.min_batch_size > config.max_batch_size:
                return {"valid": False, "error": "min_batch_size cannot be greater than max_batch_size"}
            
            if config.high_watermark <= config.low_watermark:
                return {"valid": False, "error": "high_watermark must be greater than low_watermark"}
            
            return {"valid": True}
            
        except Exception as e:
            return {"valid": False, "error": f"Validation error: {str(e)}"}
    
    async def _process_stream(
        self, stream_id: str, configuration: StreamingConfiguration
    ) -> None:
        """Main processing loop for streaming detection."""
        try:
            logger.info(f"Starting processing loop for stream {stream_id}")
            
            while stream_id in self._active_streams:
                stream_state = self._active_streams[stream_id]
                
                # Check circuit breaker
                if stream_state["circuit_breaker_open"]:
                    await asyncio.sleep(0.1)  # Wait before retrying
                    continue
                
                # Get batch of samples to process
                batch = await self._get_processing_batch(stream_id, configuration)
                
                if not batch:
                    # No samples to process, wait briefly
                    await asyncio.sleep(configuration.batch_timeout_ms / 1000.0)
                    continue
                
                # Process batch
                results = await self._process_batch(stream_id, batch, configuration)
                
                # Store results
                if results and configuration.enable_result_buffering:
                    result_buffer = stream_state["result_buffer"]
                    for result in results:
                        result_buffer.append(result)
                
                # Update metrics
                await self._update_processing_metrics(stream_id, batch, results)
                
                # Brief pause to prevent CPU spinning
                await asyncio.sleep(0.001)
                
        except asyncio.CancelledError:
            logger.info(f"Processing loop cancelled for stream {stream_id}")
        except Exception as e:
            logger.error(f"Error in processing loop for stream {stream_id}: {str(e)}")
    
    async def _get_processing_batch(
        self, stream_id: str, configuration: StreamingConfiguration
    ) -> List[StreamingSample]:
        """Get batch of samples for processing."""
        try:
            stream_state = self._active_streams[stream_id]
            buffer = stream_state["buffer"]
            
            if not buffer:
                return []
            
            # Determine batch size based on strategy
            if configuration.strategy == StreamingStrategy.REAL_TIME:
                batch_size = 1
            elif configuration.strategy == StreamingStrategy.MICRO_BATCH:
                batch_size = min(configuration.min_batch_size, len(buffer))
            elif configuration.strategy == StreamingStrategy.ADAPTIVE_BATCH:
                # Adaptive batch sizing based on buffer utilization
                utilization = len(buffer) / configuration.max_buffer_size
                if utilization > 0.8:
                    batch_size = configuration.max_batch_size
                elif utilization > 0.5:
                    batch_size = configuration.max_batch_size // 2
                else:
                    batch_size = configuration.min_batch_size
                batch_size = min(batch_size, len(buffer))
            else:
                batch_size = min(configuration.max_batch_size, len(buffer))
            
            # Extract batch
            batch = []
            for _ in range(batch_size):
                if buffer:
                    batch.append(buffer.popleft())
            
            return batch
            
        except Exception as e:
            logger.error(f"Error getting processing batch for stream {stream_id}: {str(e)}")
            return []
    
    async def _process_batch(
        self, stream_id: str, batch: List[StreamingSample], configuration: StreamingConfiguration
    ) -> List[StreamingResult]:
        """Process a batch of samples."""
        try:
            stream_state = self._active_streams[stream_id]
            request = stream_state["request"]
            
            # Get detector and adapter
            detector = await self.detector_repository.get(request.detector_id)
            adapter = self.adapter_registry.get_adapter(detector.algorithm.lower())
            
            results = []
            
            for sample in batch:
                start_time = time.time()
                
                # Prepare data
                if isinstance(sample.data, dict):
                    # Convert dict to array
                    data_array = np.array([list(sample.data.values())])
                elif isinstance(sample.data, np.ndarray):
                    if sample.data.ndim == 1:
                        data_array = sample.data.reshape(1, -1)
                    else:
                        data_array = sample.data
                else:
                    logger.warning(f"Unsupported data type for sample {sample.id}")
                    continue
                
                # Perform detection
                try:
                    predictions, scores = adapter.predict(detector, data_array)
                    
                    prediction = int(predictions[0])
                    anomaly_score = float(scores[0])
                    
                    # Calculate confidence (simplified)
                    confidence = abs(anomaly_score - 0.5) * 2
                    
                    processing_time = time.time() - start_time
                    
                    result = StreamingResult(
                        sample_id=sample.id,
                        prediction=prediction,
                        anomaly_score=anomaly_score,
                        confidence=confidence,
                        processing_time=processing_time,
                        detector_id=request.detector_id,
                        metadata=sample.metadata
                    )
                    
                    results.append(result)
                    
                    # Update anomaly count
                    if prediction == 1:
                        self._stream_metrics[stream_id].anomalies_detected += 1
                    
                except Exception as e:
                    logger.error(f"Error processing sample {sample.id}: {str(e)}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing batch for stream {stream_id}: {str(e)}")
            return []
    
    async def _update_processing_metrics(
        self, stream_id: str, batch: List[StreamingSample], results: List[StreamingResult]
    ) -> None:
        """Update processing metrics for stream."""
        try:
            metrics = self._stream_metrics[stream_id]
            stream_state = self._active_streams[stream_id]
            
            # Update sample count
            metrics.samples_processed += len(batch)
            
            # Update processing times
            if results:
                processing_times = [r.processing_time for r in results]
                stream_state["processing_stats"]["processing_times"].extend(processing_times)
                
                # Calculate average processing time
                all_times = list(stream_state["processing_stats"]["processing_times"])
                metrics.average_processing_time = np.mean(all_times)
            
            # Calculate throughput
            elapsed_time = time.time() - stream_state["processing_stats"]["start_time"]
            if elapsed_time > 0:
                metrics.throughput_per_second = metrics.samples_processed / elapsed_time
            
            # Update timestamp
            metrics.last_updated = time.time()
            
        except Exception as e:
            logger.error(f"Error updating metrics for stream {stream_id}: {str(e)}")
    
    async def _monitor_stream(self, stream_id: str) -> None:
        """Monitor stream health and performance."""
        try:
            while stream_id in self._active_streams:
                # Update buffer utilization
                stream_state = self._active_streams[stream_id]
                config = stream_state["request"].configuration
                buffer = stream_state["buffer"]
                
                utilization = len(buffer) / config.max_buffer_size
                self._stream_metrics[stream_id].buffer_utilization = utilization
                self._stream_metrics[stream_id].current_buffer_size = len(buffer)
                
                # Quality monitoring
                if config.enable_quality_monitoring:
                    await self._perform_quality_check(stream_id)
                
                await asyncio.sleep(1.0)  # Monitor every second
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error monitoring stream {stream_id}: {str(e)}")
    
    async def _perform_quality_check(self, stream_id: str) -> None:
        """Perform quality check on stream."""
        try:
            # Simplified quality check - in production would be more sophisticated
            metrics = self._stream_metrics[stream_id]
            
            # Check error rate
            total_samples = metrics.samples_processed + metrics.samples_dropped
            if total_samples > 0:
                error_rate = metrics.samples_dropped / total_samples
                metrics.error_rate = error_rate
                
                # Update quality score based on error rate
                metrics.quality_score = max(0.0, 1.0 - error_rate)
            
        except Exception as e:
            logger.error(f"Error performing quality check for stream {stream_id}: {str(e)}")
    
    def _get_initial_metrics(self) -> Dict[str, Any]:
        """Get initial performance metrics."""
        return {
            "active_streams": len(self._active_streams),
            "max_concurrent_streams": self.max_concurrent_streams,
            "total_streams_created": len(self._stream_metrics),
            "distributed_processing_enabled": self.enable_distributed_processing
        }