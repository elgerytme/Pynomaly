"""Streaming anomaly detection service with backpressure control and real-time processing."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Union
from uuid import UUID, uuid4

import numpy as np

from pynomaly.domain.entities.dataset import Dataset
from pynomaly.domain.entities.detection_result import DetectionResult
from pynomaly.domain.entities.detector import Detector
from pynomaly.domain.exceptions import StreamingError, ValidationError
from pynomaly.domain.value_objects.score import Score


class StreamState(Enum):
    """Stream processing state."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"
    BACKPRESSURE = "backpressure"


class BackpressureStrategy(Enum):
    """Backpressure handling strategies."""

    DROP_OLDEST = "drop_oldest"
    DROP_NEWEST = "drop_newest"
    PAUSE_STREAM = "pause_stream"
    BATCH_COMPRESS = "batch_compress"
    ALERT_ONLY = "alert_only"


@dataclass
class StreamingConfig:
    """Configuration for streaming detection."""

    # Processing configuration
    batch_size: int = 100
    max_queue_size: int = 10000
    processing_timeout_seconds: float = 5.0

    # Backpressure configuration
    backpressure_threshold: float = 0.8  # Queue utilization threshold
    backpressure_strategy: BackpressureStrategy = BackpressureStrategy.DROP_OLDEST
    backpressure_recovery_threshold: float = 0.5

    # Performance configuration
    max_concurrent_batches: int = 5
    enable_adaptive_batching: bool = True
    adaptive_batch_min_size: int = 10
    adaptive_batch_max_size: int = 1000

    # Drift detection
    enable_drift_detection: bool = True
    drift_window_size: int = 1000
    drift_threshold: float = 0.1

    # Quality control
    enable_quality_checks: bool = True
    max_processing_latency_ms: float = 1000.0
    min_throughput_samples_per_second: float = 10.0

    # Persistence
    persist_results: bool = True
    result_buffer_size: int = 1000

    # Monitoring
    enable_metrics: bool = True
    metrics_window_size: int = 100


@dataclass
class StreamSample:
    """A single sample in the stream."""

    id: str
    data: np.ndarray
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = "unknown"
    sequence_number: Optional[int] = None


@dataclass
class StreamBatch:
    """A batch of stream samples."""

    id: str
    samples: List[StreamSample]
    created_at: datetime
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.samples)

    @property
    def data_matrix(self) -> np.ndarray:
        """Get data as a matrix for batch processing."""
        if not self.samples:
            return np.array([])
        return np.vstack([sample.data.reshape(1, -1) for sample in self.samples])


@dataclass
class StreamingMetrics:
    """Real-time metrics for stream processing."""

    # Throughput metrics
    samples_processed: int = 0
    batches_processed: int = 0
    samples_per_second: float = 0.0
    batches_per_second: float = 0.0

    # Latency metrics
    avg_processing_latency_ms: float = 0.0
    max_processing_latency_ms: float = 0.0
    p95_processing_latency_ms: float = 0.0

    # Queue metrics
    current_queue_size: int = 0
    max_queue_size_reached: int = 0
    queue_utilization: float = 0.0

    # Backpressure metrics
    backpressure_events: int = 0
    samples_dropped: int = 0
    pause_duration_seconds: float = 0.0

    # Quality metrics
    error_rate: float = 0.0
    anomaly_rate: float = 0.0
    processing_errors: int = 0

    # Resource usage
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0

    # Drift metrics
    drift_detected: bool = False
    drift_score: float = 0.0
    model_retrain_needed: bool = False

    # Timestamps
    last_updated: datetime = field(default_factory=datetime.utcnow)
    stream_start_time: datetime = field(default_factory=datetime.utcnow)


@dataclass
class StreamingResult:
    """Result of streaming detection for a batch."""

    batch_id: str
    sample_results: List[DetectionResult]
    processing_time_ms: float
    anomalies_detected: int
    batch_size: int
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class StreamingDetectionService:
    """High-performance streaming anomaly detection service."""

    def __init__(
        self,
        detector: Detector,
        config: Optional[StreamingConfig] = None,
        result_handler: Optional[Callable[[StreamingResult], None]] = None,
        anomaly_callback: Optional[Callable[[DetectionResult], None]] = None,
    ):
        """Initialize streaming detection service.

        Args:
            detector: Trained anomaly detector
            config: Streaming configuration
            result_handler: Callback for processing results
            anomaly_callback: Callback for anomaly alerts
        """
        self.detector = detector
        self.config = config or StreamingConfig()
        self.result_handler = result_handler
        self.anomaly_callback = anomaly_callback
        self.logger = logging.getLogger(__name__)

        # Stream state
        self.state = StreamState.IDLE
        self._stop_event = asyncio.Event()
        self._pause_event = asyncio.Event()

        # Processing queues
        self.input_queue: asyncio.Queue = asyncio.Queue(
            maxsize=self.config.max_queue_size
        )
        self.batch_queue: asyncio.Queue = asyncio.Queue(
            maxsize=self.config.max_concurrent_batches
        )
        self.result_queue: asyncio.Queue = asyncio.Queue(
            maxsize=self.config.result_buffer_size
        )

        # Metrics and monitoring
        self.metrics = StreamingMetrics()
        self._metrics_history = deque(maxlen=self.config.metrics_window_size)
        self._latency_history = deque(maxlen=1000)

        # Drift detection
        self._drift_window = deque(maxlen=self.config.drift_window_size)
        self._reference_distribution: Optional[np.ndarray] = None

        # Adaptive batching
        self._current_batch_size = self.config.batch_size
        self._last_throughput_measurement = time.time()
        self._throughput_samples = deque(maxlen=10)

        # Processing tasks
        self._processing_tasks: List[asyncio.Task] = []

        # Sample sequence counter
        self._sequence_counter = 0

        # Detection result callbacks
        self._detection_callbacks: List[Callable[[StreamingResult], None]] = []

    async def start_stream(self) -> None:
        """Start the streaming detection service."""

        if self.state != StreamState.IDLE:
            raise StreamingError(f"Cannot start stream in state: {self.state}")

        self.logger.info("Starting streaming detection service")

        try:
            self.state = StreamState.RUNNING
            self.metrics.stream_start_time = datetime.utcnow()

            # Start processing tasks
            self._processing_tasks = [
                asyncio.create_task(self._batch_creation_loop()),
                asyncio.create_task(self._batch_processing_loop()),
                asyncio.create_task(self._result_handling_loop()),
                asyncio.create_task(self._metrics_update_loop()),
            ]

            # Start drift detection if enabled
            if self.config.enable_drift_detection:
                self._processing_tasks.append(
                    asyncio.create_task(self._drift_detection_loop())
                )

            # Start adaptive batching if enabled
            if self.config.enable_adaptive_batching:
                self._processing_tasks.append(
                    asyncio.create_task(self._adaptive_batching_loop())
                )

            self.logger.info("Streaming detection service started successfully")

        except Exception as e:
            self.state = StreamState.ERROR
            self.logger.error(f"Failed to start streaming service: {e}")
            raise StreamingError(f"Stream start failed: {e}")

    async def stop_stream(self) -> None:
        """Stop the streaming detection service gracefully."""

        self.logger.info("Stopping streaming detection service")

        # Signal stop
        self._stop_event.set()
        self.state = StreamState.STOPPED

        # Cancel all processing tasks
        for task in self._processing_tasks:
            if not task.done():
                task.cancel()

        # Wait for tasks to complete
        if self._processing_tasks:
            await asyncio.gather(*self._processing_tasks, return_exceptions=True)

        # Process remaining items in queues
        await self._drain_queues()

        self.logger.info("Streaming detection service stopped")

    async def pause_stream(self) -> None:
        """Pause stream processing."""

        if self.state != StreamState.RUNNING:
            raise StreamingError(f"Cannot pause stream in state: {self.state}")

        self.state = StreamState.PAUSED
        self._pause_event.set()
        self.logger.info("Stream processing paused")

    async def resume_stream(self) -> None:
        """Resume stream processing."""

        if self.state != StreamState.PAUSED:
            raise StreamingError(f"Cannot resume stream in state: {self.state}")

        self.state = StreamState.RUNNING
        self._pause_event.clear()
        self.logger.info("Stream processing resumed")

    async def process_sample(
        self,
        data: Union[np.ndarray, List[float], Dict[str, Any]],
        sample_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        source: str = "api",
    ) -> bool:
        """Process a single sample through the stream.

        Args:
            data: Sample data
            sample_id: Optional sample ID
            metadata: Optional metadata
            source: Data source identifier

        Returns:
            True if sample was queued successfully, False if dropped due to backpressure
        """

        if self.state not in [StreamState.RUNNING, StreamState.PAUSED]:
            raise StreamingError(f"Cannot process samples in state: {self.state}")

        # Convert data to numpy array
        if isinstance(data, dict):
            # Assume data is in 'features' key or similar
            data_array = np.array(list(data.values()))
        elif isinstance(data, list):
            data_array = np.array(data)
        elif isinstance(data, np.ndarray):
            data_array = data
        else:
            raise ValidationError(f"Unsupported data type: {type(data)}")

        # Ensure 1D array
        if data_array.ndim > 1:
            data_array = data_array.flatten()

        # Create stream sample
        sample = StreamSample(
            id=sample_id or str(uuid4()),
            data=data_array,
            timestamp=datetime.utcnow(),
            metadata=metadata or {},
            source=source,
            sequence_number=self._sequence_counter,
        )
        self._sequence_counter += 1

        # Check backpressure
        if await self._check_backpressure():
            return False

        # Queue sample
        try:
            await self.input_queue.put(sample)
            return True

        except asyncio.QueueFull:
            # Apply backpressure strategy
            return await self._handle_backpressure(sample)

    async def process_batch(
        self,
        samples: List[Dict[str, Any]],
        batch_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Process a batch of samples.

        Args:
            samples: List of sample data
            batch_id: Optional batch ID
            metadata: Optional batch metadata

        Returns:
            True if batch was processed successfully
        """

        success_count = 0

        for i, sample_data in enumerate(samples):
            sample_id = f"{batch_id or 'batch'}_{i}" if batch_id else None
            success = await self.process_sample(
                data=sample_data,
                sample_id=sample_id,
                metadata=metadata,
                source="batch",
            )
            if success:
                success_count += 1

        return success_count == len(samples)

    async def get_metrics(self) -> StreamingMetrics:
        """Get current streaming metrics."""

        # Update real-time metrics
        self.metrics.current_queue_size = self.input_queue.qsize()
        self.metrics.queue_utilization = (
            self.input_queue.qsize() / self.config.max_queue_size
        )
        self.metrics.last_updated = datetime.utcnow()

        return self.metrics

    async def get_stream_health(self) -> Dict[str, Any]:
        """Get stream health status."""

        metrics = await self.get_metrics()

        # Determine health status
        health_issues = []

        if metrics.queue_utilization > 0.9:
            health_issues.append("High queue utilization")

        if metrics.error_rate > 0.05:
            health_issues.append("High error rate")

        if metrics.avg_processing_latency_ms > self.config.max_processing_latency_ms:
            health_issues.append("High processing latency")

        if metrics.samples_per_second < self.config.min_throughput_samples_per_second:
            health_issues.append("Low throughput")

        if metrics.drift_detected:
            health_issues.append("Data drift detected")

        # Overall health score
        health_score = max(0.0, 1.0 - len(health_issues) * 0.2)

        return {
            "state": self.state.value,
            "health_score": health_score,
            "health_status": (
                "healthy"
                if health_score > 0.8
                else "degraded" if health_score > 0.5 else "unhealthy"
            ),
            "issues": health_issues,
            "metrics": metrics,
            "uptime_seconds": (
                datetime.utcnow() - metrics.stream_start_time
            ).total_seconds(),
        }

    def register_detection_callback(
        self, callback: Callable[[StreamingResult], None]
    ) -> None:
        """Register a callback for detection results.

        Args:
            callback: Function to call when detection results are available
        """
        if callback not in self._detection_callbacks:
            self._detection_callbacks.append(callback)
            self.logger.info(f"Registered detection callback: {callback.__name__}")

    def unregister_detection_callback(
        self, callback: Callable[[StreamingResult], None]
    ) -> None:
        """Unregister a detection result callback.

        Args:
            callback: Function to remove from callbacks
        """
        if callback in self._detection_callbacks:
            self._detection_callbacks.remove(callback)
            self.logger.info(f"Unregistered detection callback: {callback.__name__}")

    async def _notify_detection_callbacks(self, result: StreamingResult) -> None:
        """Notify all registered callbacks of detection results.

        Args:
            result: Detection result to send to callbacks
        """
        for callback in self._detection_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(result)
                else:
                    callback(result)
            except Exception as e:
                self.logger.error(
                    f"Error in detection callback {callback.__name__}: {e}"
                )

    async def _batch_creation_loop(self) -> None:
        """Main loop for creating batches from individual samples."""

        current_batch_samples = []
        batch_timeout = asyncio.create_task(asyncio.sleep(1.0))  # 1 second timeout

        while not self._stop_event.is_set():
            try:
                # Wait for either a sample or timeout
                done, pending = await asyncio.wait(
                    [
                        asyncio.create_task(self.input_queue.get()),
                        batch_timeout,
                    ],
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=0.1,
                )

                # Cancel pending tasks
                for task in pending:
                    task.cancel()

                # Check if we got a sample
                sample = None
                timeout_occurred = False

                for task in done:
                    if hasattr(task, "_coro") and "get" in str(task._coro):
                        sample = await task
                    else:
                        timeout_occurred = True

                # Add sample to current batch
                if sample is not None:
                    current_batch_samples.append(sample)

                # Check if we should create a batch
                should_create_batch = (
                    len(current_batch_samples) >= self._current_batch_size
                    or (timeout_occurred and current_batch_samples)
                    or len(current_batch_samples) >= self.config.adaptive_batch_max_size
                )

                if should_create_batch and current_batch_samples:
                    # Create batch
                    batch = StreamBatch(
                        id=str(uuid4()),
                        samples=current_batch_samples.copy(),
                        created_at=datetime.utcnow(),
                    )

                    # Queue batch for processing
                    try:
                        await self.batch_queue.put(batch)
                        current_batch_samples.clear()

                    except asyncio.QueueFull:
                        self.logger.warning("Batch queue full, dropping batch")
                        current_batch_samples.clear()

                # Reset timeout if it occurred
                if timeout_occurred or not batch_timeout.done():
                    if not batch_timeout.done():
                        batch_timeout.cancel()
                    batch_timeout = asyncio.create_task(asyncio.sleep(1.0))

                # Handle pause
                if self._pause_event.is_set():
                    await asyncio.sleep(0.1)

            except Exception as e:
                self.logger.error(f"Error in batch creation loop: {e}")
                await asyncio.sleep(0.1)

    async def _batch_processing_loop(self) -> None:
        """Main loop for processing batches."""

        while not self._stop_event.is_set():
            try:
                # Get batch from queue
                try:
                    batch = await asyncio.wait_for(self.batch_queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue

                # Skip processing if paused
                if self._pause_event.is_set():
                    # Put batch back in queue
                    try:
                        await self.batch_queue.put(batch)
                    except asyncio.QueueFull:
                        pass
                    await asyncio.sleep(0.1)
                    continue

                # Process batch
                start_time = time.time()

                try:
                    result = await self._process_batch(batch)
                    processing_time = (time.time() - start_time) * 1000  # Convert to ms

                    # Update metrics
                    self.metrics.batches_processed += 1
                    self.metrics.samples_processed += batch.size
                    self._latency_history.append(processing_time)

                    # Update latency metrics
                    if self._latency_history:
                        self.metrics.avg_processing_latency_ms = np.mean(
                            self._latency_history
                        )
                        self.metrics.max_processing_latency_ms = max(
                            self._latency_history
                        )
                        self.metrics.p95_processing_latency_ms = np.percentile(
                            self._latency_history, 95
                        )

                    # Queue result
                    await self.result_queue.put(result)

                except Exception as e:
                    self.logger.error(f"Error processing batch {batch.id}: {e}")
                    self.metrics.processing_errors += 1

            except Exception as e:
                self.logger.error(f"Error in batch processing loop: {e}")
                await asyncio.sleep(0.1)

    async def _result_handling_loop(self) -> None:
        """Main loop for handling processing results."""

        while not self._stop_event.is_set():
            try:
                # Get result from queue
                try:
                    result = await asyncio.wait_for(
                        self.result_queue.get(), timeout=0.1
                    )
                except asyncio.TimeoutError:
                    continue

                # Handle result
                try:
                    # Call result handler if provided
                    if self.result_handler:
                        await asyncio.create_task(
                            self._safe_call_handler(self.result_handler, result)
                        )

                    # Notify detection callbacks
                    await self._notify_detection_callbacks(result)

                    # Handle individual anomalies
                    if self.anomaly_callback:
                        for sample_result in result.sample_results:
                            if sample_result.is_anomaly:
                                await asyncio.create_task(
                                    self._safe_call_handler(
                                        self.anomaly_callback, sample_result
                                    )
                                )

                    # Update anomaly rate
                    if result.batch_size > 0:
                        batch_anomaly_rate = (
                            result.anomalies_detected / result.batch_size
                        )
                        # Exponential moving average
                        alpha = 0.1
                        self.metrics.anomaly_rate = (
                            alpha * batch_anomaly_rate
                            + (1 - alpha) * self.metrics.anomaly_rate
                        )

                except Exception as e:
                    self.logger.error(f"Error handling result: {e}")
                    self.metrics.processing_errors += 1

            except Exception as e:
                self.logger.error(f"Error in result handling loop: {e}")
                await asyncio.sleep(0.1)

    async def _metrics_update_loop(self) -> None:
        """Main loop for updating metrics."""

        last_update = time.time()
        last_samples = 0
        last_batches = 0

        while not self._stop_event.is_set():
            try:
                await asyncio.sleep(1.0)  # Update every second

                current_time = time.time()
                time_diff = current_time - last_update

                if time_diff > 0:
                    # Calculate throughput
                    samples_diff = self.metrics.samples_processed - last_samples
                    batches_diff = self.metrics.batches_processed - last_batches

                    self.metrics.samples_per_second = samples_diff / time_diff
                    self.metrics.batches_per_second = batches_diff / time_diff

                    # Update error rate
                    if self.metrics.samples_processed > 0:
                        self.metrics.error_rate = (
                            self.metrics.processing_errors
                            / self.metrics.samples_processed
                        )

                    # Store in history
                    self._metrics_history.append(
                        {
                            "timestamp": datetime.utcnow(),
                            "samples_per_second": self.metrics.samples_per_second,
                            "queue_utilization": self.metrics.queue_utilization,
                            "avg_latency": self.metrics.avg_processing_latency_ms,
                            "error_rate": self.metrics.error_rate,
                        }
                    )

                    # Update for next iteration
                    last_update = current_time
                    last_samples = self.metrics.samples_processed
                    last_batches = self.metrics.batches_processed

            except Exception as e:
                self.logger.error(f"Error in metrics update loop: {e}")

    async def _drift_detection_loop(self) -> None:
        """Main loop for drift detection."""

        while not self._stop_event.is_set():
            try:
                await asyncio.sleep(10.0)  # Check every 10 seconds

                if len(self._drift_window) >= self.config.drift_window_size:
                    # Perform drift detection
                    drift_detected = await self._detect_drift()
                    self.metrics.drift_detected = drift_detected

                    if drift_detected:
                        self.logger.warning("Data drift detected in stream")
                        self.metrics.model_retrain_needed = True

            except Exception as e:
                self.logger.error(f"Error in drift detection loop: {e}")

    async def _adaptive_batching_loop(self) -> None:
        """Main loop for adaptive batch size adjustment."""

        while not self._stop_event.is_set():
            try:
                await asyncio.sleep(5.0)  # Adjust every 5 seconds

                # Get recent throughput measurements
                if len(self._throughput_samples) >= 3:
                    avg_throughput = np.mean(self._throughput_samples)
                    target_throughput = (
                        self.config.min_throughput_samples_per_second * 2
                    )

                    # Adjust batch size based on throughput
                    if avg_throughput < target_throughput * 0.8:
                        # Increase batch size to improve throughput
                        self._current_batch_size = min(
                            self._current_batch_size + 10,
                            self.config.adaptive_batch_max_size,
                        )
                    elif avg_throughput > target_throughput * 1.2:
                        # Decrease batch size to reduce latency
                        self._current_batch_size = max(
                            self._current_batch_size - 10,
                            self.config.adaptive_batch_min_size,
                        )

                    # Store current throughput
                    self._throughput_samples.append(self.metrics.samples_per_second)

            except Exception as e:
                self.logger.error(f"Error in adaptive batching loop: {e}")

    async def _process_batch(self, batch: StreamBatch) -> StreamingResult:
        """Process a single batch of samples."""

        start_time = time.time()

        try:
            # Get data matrix
            data_matrix = batch.data_matrix

            if data_matrix.size == 0:
                return StreamingResult(
                    batch_id=batch.id,
                    sample_results=[],
                    processing_time_ms=0.0,
                    anomalies_detected=0,
                    batch_size=0,
                )

            # Add to drift detection window
            if self.config.enable_drift_detection:
                for sample in batch.samples:
                    self._drift_window.append(sample.data)

            # Perform anomaly detection (placeholder)
            # In practice, this would use the actual detector
            scores = np.random.random(len(batch.samples))
            predictions = (scores > 0.8).astype(int)

            # Create individual results
            sample_results = []
            anomalies_detected = 0

            for i, (sample, score, prediction) in enumerate(
                zip(batch.samples, scores, predictions)
            ):
                is_anomaly = bool(prediction)
                if is_anomaly:
                    anomalies_detected += 1

                result = DetectionResult(
                    id=uuid4(),
                    detector_id=self.detector.id,
                    dataset_id=uuid4(),  # Would be actual dataset ID
                    scores=[Score(value=float(score), confidence=0.8)],
                    is_anomaly=is_anomaly,
                    anomaly_threshold=0.8,
                    n_anomalies=1 if is_anomaly else 0,
                    execution_time_ms=(time.time() - start_time) * 1000,
                    timestamp=sample.timestamp,
                    metadata={
                        **sample.metadata,
                        "sample_id": sample.id,
                        "source": sample.source,
                        "sequence_number": sample.sequence_number,
                    },
                )
                sample_results.append(result)

            processing_time = (time.time() - start_time) * 1000

            return StreamingResult(
                batch_id=batch.id,
                sample_results=sample_results,
                processing_time_ms=processing_time,
                anomalies_detected=anomalies_detected,
                batch_size=batch.size,
                metadata=batch.metadata,
            )

        except Exception as e:
            self.logger.error(f"Error processing batch {batch.id}: {e}")
            raise StreamingError(f"Batch processing failed: {e}")

    async def _check_backpressure(self) -> bool:
        """Check if backpressure conditions are met."""

        queue_utilization = self.input_queue.qsize() / self.config.max_queue_size

        if queue_utilization >= self.config.backpressure_threshold:
            self.metrics.backpressure_events += 1

            if self.state != StreamState.BACKPRESSURE:
                self.logger.warning(
                    f"Backpressure threshold reached: {queue_utilization:.2f}"
                )
                self.state = StreamState.BACKPRESSURE

            return True

        # Check for recovery from backpressure
        if (
            self.state == StreamState.BACKPRESSURE
            and queue_utilization <= self.config.backpressure_recovery_threshold
        ):
            self.logger.info("Recovered from backpressure")
            self.state = StreamState.RUNNING

        return False

    async def _handle_backpressure(self, sample: StreamSample) -> bool:
        """Handle backpressure according to configured strategy."""

        strategy = self.config.backpressure_strategy

        if strategy == BackpressureStrategy.DROP_NEWEST:
            # Drop the new sample
            self.metrics.samples_dropped += 1
            self.logger.debug(f"Dropped new sample {sample.id} due to backpressure")
            return False

        elif strategy == BackpressureStrategy.DROP_OLDEST:
            # Drop oldest sample and queue new one
            try:
                if not self.input_queue.empty():
                    old_sample = await asyncio.wait_for(
                        self.input_queue.get(), timeout=0.01
                    )
                    self.metrics.samples_dropped += 1
                    self.logger.debug(
                        f"Dropped old sample {old_sample.id} due to backpressure"
                    )

                await self.input_queue.put(sample)
                return True

            except (asyncio.TimeoutError, asyncio.QueueFull):
                self.metrics.samples_dropped += 1
                return False

        elif strategy == BackpressureStrategy.PAUSE_STREAM:
            # Pause stream processing
            if self.state != StreamState.PAUSED:
                await self.pause_stream()
            return False

        elif strategy == BackpressureStrategy.ALERT_ONLY:
            # Just log the backpressure but try to queue anyway
            self.logger.warning(f"Backpressure detected but continuing processing")
            try:
                await self.input_queue.put(sample)
                return True
            except asyncio.QueueFull:
                self.metrics.samples_dropped += 1
                return False

        else:
            # Default: drop the sample
            self.metrics.samples_dropped += 1
            return False

    async def _detect_drift(self) -> bool:
        """Detect if data drift has occurred."""

        if len(self._drift_window) < self.config.drift_window_size:
            return False

        try:
            # Convert to numpy array
            current_data = np.array(list(self._drift_window))

            # Set reference distribution if not set
            if self._reference_distribution is None:
                self._reference_distribution = current_data[
                    : self.config.drift_window_size // 2
                ]
                return False

            # Compare distributions using simple statistical test
            # In practice, you might use more sophisticated methods like KS test, PSI, etc.

            # Calculate mean and std differences
            ref_mean = np.mean(self._reference_distribution, axis=0)
            ref_std = np.std(self._reference_distribution, axis=0)

            current_mean = np.mean(
                current_data[-(self.config.drift_window_size // 2) :], axis=0
            )
            current_std = np.std(
                current_data[-(self.config.drift_window_size // 2) :], axis=0
            )

            # Calculate normalized differences
            mean_diff = np.mean(np.abs(current_mean - ref_mean) / (ref_std + 1e-8))
            std_diff = np.mean(np.abs(current_std - ref_std) / (ref_std + 1e-8))

            # Combine into drift score
            drift_score = (mean_diff + std_diff) / 2
            self.metrics.drift_score = float(drift_score)

            # Check if drift threshold is exceeded
            if drift_score > self.config.drift_threshold:
                # Update reference distribution
                self._reference_distribution = current_data[
                    -(self.config.drift_window_size // 2) :
                ]
                return True

            return False

        except Exception as e:
            self.logger.error(f"Error in drift detection: {e}")
            return False

    async def _safe_call_handler(self, handler: Callable, *args, **kwargs) -> None:
        """Safely call a handler function."""

        try:
            if asyncio.iscoroutinefunction(handler):
                await handler(*args, **kwargs)
            else:
                handler(*args, **kwargs)

        except Exception as e:
            self.logger.error(f"Error in handler: {e}")

    async def _drain_queues(self) -> None:
        """Drain remaining items from queues."""

        try:
            # Process remaining batches
            while not self.batch_queue.empty():
                try:
                    batch = await asyncio.wait_for(self.batch_queue.get(), timeout=0.1)
                    result = await self._process_batch(batch)

                    if self.result_handler:
                        await self._safe_call_handler(self.result_handler, result)

                except asyncio.TimeoutError:
                    break
                except Exception as e:
                    self.logger.warning(f"Error draining batch queue: {e}")

            # Process remaining results
            while not self.result_queue.empty():
                try:
                    result = await asyncio.wait_for(
                        self.result_queue.get(), timeout=0.1
                    )

                    if self.result_handler:
                        await self._safe_call_handler(self.result_handler, result)

                except asyncio.TimeoutError:
                    break
                except Exception as e:
                    self.logger.warning(f"Error draining result queue: {e}")

        except Exception as e:
            self.logger.error(f"Error draining queues: {e}")


class StreamingDetectionFactory:
    """Factory for creating streaming detection services."""

    @staticmethod
    def create_service(
        detector: Detector, config: Optional[StreamingConfig] = None, **kwargs
    ) -> StreamingDetectionService:
        """Create a streaming detection service.

        Args:
            detector: Trained detector
            config: Streaming configuration
            **kwargs: Additional arguments for service

        Returns:
            Configured streaming detection service
        """
        return StreamingDetectionService(detector, config, **kwargs)

    @staticmethod
    def create_high_throughput_config() -> StreamingConfig:
        """Create configuration optimized for high throughput."""
        return StreamingConfig(
            batch_size=500,
            max_queue_size=50000,
            max_concurrent_batches=10,
            enable_adaptive_batching=True,
            adaptive_batch_max_size=2000,
            backpressure_strategy=BackpressureStrategy.DROP_OLDEST,
            enable_drift_detection=False,  # Disable for performance
            enable_quality_checks=False,
        )

    @staticmethod
    def create_low_latency_config() -> StreamingConfig:
        """Create configuration optimized for low latency."""
        return StreamingConfig(
            batch_size=10,
            max_queue_size=1000,
            max_concurrent_batches=20,
            enable_adaptive_batching=True,
            adaptive_batch_min_size=1,
            adaptive_batch_max_size=50,
            processing_timeout_seconds=0.1,
            backpressure_strategy=BackpressureStrategy.PAUSE_STREAM,
        )

    @staticmethod
    def create_balanced_config() -> StreamingConfig:
        """Create balanced configuration for general use."""
        return StreamingConfig(
            batch_size=100,
            max_queue_size=10000,
            max_concurrent_batches=5,
            enable_adaptive_batching=True,
            enable_drift_detection=True,
            enable_quality_checks=True,
            backpressure_strategy=BackpressureStrategy.DROP_OLDEST,
        )
