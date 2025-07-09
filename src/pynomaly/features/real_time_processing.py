"""Real-time processing and streaming anomaly detection for Pynomaly."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import pandas as pd

from pynomaly.domain.entities import Dataset, DetectionResult, Detector
from pynomaly.shared.error_handling import (
    ErrorCodes,
    create_infrastructure_error,
)

logger = logging.getLogger(__name__)


class StreamEventType(Enum):
    """Types of streaming events."""

    DATA_POINT = "data_point"
    ANOMALY_DETECTED = "anomaly_detected"
    SYSTEM_STATUS = "system_status"
    ERROR = "error"
    HEARTBEAT = "heartbeat"


class ProcessingMode(Enum):
    """Stream processing modes."""

    REAL_TIME = "real_time"
    BATCH = "batch"
    MICRO_BATCH = "micro_batch"


@dataclass
class StreamEvent:
    """Streaming event data structure."""

    event_id: str
    event_type: StreamEventType
    timestamp: datetime
    data: dict[str, Any]
    source: str = ""
    correlation_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "source": self.source,
            "correlation_id": self.correlation_id,
        }


@dataclass
class StreamingConfig:
    """Configuration for streaming processing."""

    buffer_size: int = 1000
    batch_size: int = 100
    batch_timeout_seconds: float = 5.0
    processing_mode: ProcessingMode = ProcessingMode.REAL_TIME
    enable_backpressure: bool = True
    max_memory_mb: int = 512
    heartbeat_interval_seconds: float = 30.0
    error_threshold: int = 100
    window_size_minutes: int = 60

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "buffer_size": self.buffer_size,
            "batch_size": self.batch_size,
            "batch_timeout_seconds": self.batch_timeout_seconds,
            "processing_mode": self.processing_mode.value,
            "enable_backpressure": self.enable_backpressure,
            "max_memory_mb": self.max_memory_mb,
            "heartbeat_interval_seconds": self.heartbeat_interval_seconds,
            "error_threshold": self.error_threshold,
            "window_size_minutes": self.window_size_minutes,
        }


class StreamBuffer:
    """Thread-safe buffer for streaming data."""

    def __init__(self, max_size: int = 1000):
        """Initialize stream buffer.

        Args:
            max_size: Maximum buffer size
        """
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.lock = asyncio.Lock()
        self.stats = {
            "total_received": 0,
            "total_processed": 0,
            "buffer_overflows": 0,
            "last_update": datetime.now(),
        }

    async def add(self, event: StreamEvent) -> bool:
        """Add event to buffer.

        Args:
            event: Stream event to add

        Returns:
            True if added successfully
        """
        async with self.lock:
            if len(self.buffer) >= self.max_size:
                self.stats["buffer_overflows"] += 1
                logger.warning(f"Buffer overflow: dropping event {event.event_id}")
                return False

            self.buffer.append(event)
            self.stats["total_received"] += 1
            self.stats["last_update"] = datetime.now()
            return True

    async def get_batch(self, batch_size: int) -> list[StreamEvent]:
        """Get batch of events from buffer.

        Args:
            batch_size: Number of events to retrieve

        Returns:
            List of stream events
        """
        async with self.lock:
            batch = []
            for _ in range(min(batch_size, len(self.buffer))):
                if self.buffer:
                    batch.append(self.buffer.popleft())

            self.stats["total_processed"] += len(batch)
            return batch

    async def size(self) -> int:
        """Get current buffer size."""
        async with self.lock:
            return len(self.buffer)

    async def is_empty(self) -> bool:
        """Check if buffer is empty."""
        async with self.lock:
            return len(self.buffer) == 0

    async def get_stats(self) -> dict[str, Any]:
        """Get buffer statistics."""
        async with self.lock:
            return {
                **self.stats,
                "current_size": len(self.buffer),
                "max_size": self.max_size,
                "utilization": len(self.buffer) / self.max_size,
            }


class RealTimeDetector:
    """Real-time anomaly detector with streaming capabilities."""

    def __init__(self, detector: Detector, config: StreamingConfig):
        """Initialize real-time detector.

        Args:
            detector: Base detector instance
            config: Streaming configuration
        """
        self.detector = detector
        self.config = config
        self.is_running = False
        self.sliding_window = deque(
            maxlen=config.window_size_minutes * 60
        )  # 1 sample per second
        self.detection_stats = {
            "total_detections": 0,
            "anomalies_detected": 0,
            "false_positives": 0,
            "processing_time_ms": [],
            "last_detection": None,
        }

    async def detect_streaming(
        self, data_point: dict[str, Any]
    ) -> DetectionResult | None:
        """Detect anomalies in real-time data point.

        Args:
            data_point: Single data point to analyze

        Returns:
            Detection result if anomaly detected
        """
        try:
            start_time = datetime.now()

            # Add to sliding window
            self.sliding_window.append(
                {
                    "timestamp": start_time,
                    "data": data_point,
                }
            )

            # Convert to DataFrame for detection
            df = pd.DataFrame([data_point])
            dataset = Dataset(
                name="streaming_data",
                data=df,
                description="Real-time streaming data point",
            )

            # Perform detection
            result = await self._detect_with_context(dataset)

            # Update statistics
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self.detection_stats["total_detections"] += 1
            self.detection_stats["processing_time_ms"].append(processing_time)
            self.detection_stats["last_detection"] = start_time

            # Keep only last 1000 processing times
            if len(self.detection_stats["processing_time_ms"]) > 1000:
                self.detection_stats["processing_time_ms"] = self.detection_stats[
                    "processing_time_ms"
                ][-1000:]

            if result and result.anomalies:
                self.detection_stats["anomalies_detected"] += 1
                logger.info(
                    f"Real-time anomaly detected: {len(result.anomalies)} anomalies"
                )

            return result

        except Exception as e:
            logger.error(f"Real-time detection failed: {e}")
            raise create_infrastructure_error(
                error_code=ErrorCodes.INF_PROCESSING_ERROR,
                message=f"Real-time detection failed: {str(e)}",
                cause=e,
            )

    async def _detect_with_context(self, dataset: Dataset) -> DetectionResult | None:
        """Detect anomalies with sliding window context."""
        try:
            # Use base detector
            if hasattr(self.detector, "detect"):
                return await self.detector.detect(dataset)
            else:
                # Fallback for detectors without async detect
                return self.detector.predict(dataset)

        except Exception as e:
            logger.error(f"Detection with context failed: {e}")
            return None

    async def get_detection_stats(self) -> dict[str, Any]:
        """Get detection statistics."""
        processing_times = self.detection_stats["processing_time_ms"]

        stats = {
            "total_detections": self.detection_stats["total_detections"],
            "anomalies_detected": self.detection_stats["anomalies_detected"],
            "false_positives": self.detection_stats["false_positives"],
            "anomaly_rate": (
                self.detection_stats["anomalies_detected"]
                / self.detection_stats["total_detections"]
                if self.detection_stats["total_detections"] > 0
                else 0
            ),
            "window_size": len(self.sliding_window),
            "processing_performance": {
                "avg_time_ms": sum(processing_times) / len(processing_times)
                if processing_times
                else 0,
                "min_time_ms": min(processing_times) if processing_times else 0,
                "max_time_ms": max(processing_times) if processing_times else 0,
                "samples": len(processing_times),
            },
            "last_detection": self.detection_stats["last_detection"].isoformat()
            if self.detection_stats["last_detection"]
            else None,
        }

        return stats


class EventProcessor:
    """Event processor for handling streaming events."""

    def __init__(self, config: StreamingConfig):
        """Initialize event processor.

        Args:
            config: Streaming configuration
        """
        self.config = config
        self.event_handlers: dict[StreamEventType, list[Callable]] = {}
        self.processing_stats = {
            "events_processed": 0,
            "events_failed": 0,
            "processing_errors": [],
            "start_time": datetime.now(),
        }

    def register_handler(
        self, event_type: StreamEventType, handler: Callable[[StreamEvent], None]
    ) -> None:
        """Register event handler.

        Args:
            event_type: Type of event to handle
            handler: Handler function
        """
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []

        self.event_handlers[event_type].append(handler)
        logger.info(f"Registered handler for {event_type.value} events")

    async def process_event(self, event: StreamEvent) -> bool:
        """Process single event.

        Args:
            event: Event to process

        Returns:
            True if processed successfully
        """
        try:
            handlers = self.event_handlers.get(event.event_type, [])

            if not handlers:
                logger.warning(
                    f"No handlers registered for event type: {event.event_type.value}"
                )
                return False

            # Process with all registered handlers
            for handler in handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                except Exception as e:
                    logger.error(f"Handler failed for event {event.event_id}: {e}")
                    self.processing_stats["events_failed"] += 1
                    self.processing_stats["processing_errors"].append(
                        {
                            "event_id": event.event_id,
                            "error": str(e),
                            "timestamp": datetime.now().isoformat(),
                        }
                    )

            self.processing_stats["events_processed"] += 1
            return True

        except Exception as e:
            logger.error(f"Event processing failed: {e}")
            self.processing_stats["events_failed"] += 1
            return False

    async def process_batch(self, events: list[StreamEvent]) -> dict[str, int]:
        """Process batch of events.

        Args:
            events: Events to process

        Returns:
            Processing results
        """
        results = {"success": 0, "failed": 0}

        for event in events:
            success = await self.process_event(event)
            if success:
                results["success"] += 1
            else:
                results["failed"] += 1

        return results

    async def get_processing_stats(self) -> dict[str, Any]:
        """Get processing statistics."""
        uptime = (datetime.now() - self.processing_stats["start_time"]).total_seconds()

        return {
            **self.processing_stats,
            "uptime_seconds": uptime,
            "events_per_second": self.processing_stats["events_processed"] / uptime
            if uptime > 0
            else 0,
            "error_rate": (
                self.processing_stats["events_failed"]
                / (
                    self.processing_stats["events_processed"]
                    + self.processing_stats["events_failed"]
                )
                if (
                    self.processing_stats["events_processed"]
                    + self.processing_stats["events_failed"]
                )
                > 0
                else 0
            ),
            "recent_errors": self.processing_stats["processing_errors"][
                -10:
            ],  # Last 10 errors
        }


class StreamingPipeline:
    """Complete streaming pipeline for real-time anomaly detection."""

    def __init__(self, detector: Detector, config: StreamingConfig | None = None):
        """Initialize streaming pipeline.

        Args:
            detector: Anomaly detector
            config: Streaming configuration
        """
        self.config = config or StreamingConfig()
        self.detector = RealTimeDetector(detector, self.config)
        self.buffer = StreamBuffer(self.config.buffer_size)
        self.event_processor = EventProcessor(self.config)
        self.is_running = False
        self.processing_task: asyncio.Task | None = None
        self.heartbeat_task: asyncio.Task | None = None

        # Register default handlers
        self._register_default_handlers()

    def _register_default_handlers(self) -> None:
        """Register default event handlers."""

        async def handle_data_point(event: StreamEvent):
            """Handle incoming data point."""
            try:
                data_point = event.data
                result = await self.detector.detect_streaming(data_point)

                if result and result.anomalies:
                    # Create anomaly event
                    anomaly_event = StreamEvent(
                        event_id=str(uuid.uuid4()),
                        event_type=StreamEventType.ANOMALY_DETECTED,
                        timestamp=datetime.now(),
                        data={
                            "original_event_id": event.event_id,
                            "anomalies": [a.to_dict() for a in result.anomalies],
                            "detection_result": result.to_dict(),
                        },
                        source="real_time_detector",
                        correlation_id=event.correlation_id,
                    )

                    await self.buffer.add(anomaly_event)

            except Exception as e:
                logger.error(f"Data point handling failed: {e}")

        async def handle_anomaly_detected(event: StreamEvent):
            """Handle detected anomaly."""
            logger.warning(f"Anomaly detected: {event.data.get('anomalies', [])}")

        async def handle_system_status(event: StreamEvent):
            """Handle system status events."""
            logger.info(f"System status: {event.data}")

        async def handle_error(event: StreamEvent):
            """Handle error events."""
            logger.error(f"Stream error: {event.data}")

        # Register handlers
        self.event_processor.register_handler(
            StreamEventType.DATA_POINT, handle_data_point
        )
        self.event_processor.register_handler(
            StreamEventType.ANOMALY_DETECTED, handle_anomaly_detected
        )
        self.event_processor.register_handler(
            StreamEventType.SYSTEM_STATUS, handle_system_status
        )
        self.event_processor.register_handler(StreamEventType.ERROR, handle_error)

    async def start(self) -> bool:
        """Start streaming pipeline.

        Returns:
            True if started successfully
        """
        try:
            if self.is_running:
                logger.warning("Pipeline is already running")
                return False

            self.is_running = True

            # Start processing task
            self.processing_task = asyncio.create_task(self._processing_loop())

            # Start heartbeat task
            self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())

            logger.info("Streaming pipeline started")
            return True

        except Exception as e:
            logger.error(f"Failed to start streaming pipeline: {e}")
            self.is_running = False
            return False

    async def stop(self) -> bool:
        """Stop streaming pipeline.

        Returns:
            True if stopped successfully
        """
        try:
            self.is_running = False

            # Cancel tasks
            if self.processing_task and not self.processing_task.done():
                self.processing_task.cancel()
                try:
                    await self.processing_task
                except asyncio.CancelledError:
                    pass

            if self.heartbeat_task and not self.heartbeat_task.done():
                self.heartbeat_task.cancel()
                try:
                    await self.heartbeat_task
                except asyncio.CancelledError:
                    pass

            logger.info("Streaming pipeline stopped")
            return True

        except Exception as e:
            logger.error(f"Failed to stop streaming pipeline: {e}")
            return False

    async def submit_data(self, data_point: dict[str, Any], source: str = "") -> bool:
        """Submit data point for processing.

        Args:
            data_point: Data to process
            source: Data source identifier

        Returns:
            True if submitted successfully
        """
        if not self.is_running:
            logger.warning("Pipeline is not running")
            return False

        event = StreamEvent(
            event_id=str(uuid.uuid4()),
            event_type=StreamEventType.DATA_POINT,
            timestamp=datetime.now(),
            data=data_point,
            source=source,
        )

        return await self.buffer.add(event)

    async def _processing_loop(self) -> None:
        """Main processing loop."""
        logger.info("Processing loop started")

        try:
            while self.is_running:
                # Check if buffer has data
                if await self.buffer.is_empty():
                    await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
                    continue

                # Get batch of events
                batch = await self.buffer.get_batch(self.config.batch_size)

                if not batch:
                    continue

                # Process batch
                results = await self.event_processor.process_batch(batch)

                if results["failed"] > 0:
                    logger.warning(
                        f"Batch processing: {results['success']} success, {results['failed']} failed"
                    )

                # Check for backpressure
                if self.config.enable_backpressure:
                    buffer_size = await self.buffer.size()
                    if buffer_size > self.config.buffer_size * 0.8:
                        logger.warning(
                            f"High buffer utilization: {buffer_size}/{self.config.buffer_size}"
                        )
                        await asyncio.sleep(0.1)  # Slow down processing

        except Exception as e:
            logger.error(f"Processing loop error: {e}")
        finally:
            logger.info("Processing loop stopped")

    async def _heartbeat_loop(self) -> None:
        """Heartbeat loop for monitoring."""
        logger.info("Heartbeat loop started")

        try:
            while self.is_running:
                # Send heartbeat event
                heartbeat_event = StreamEvent(
                    event_id=str(uuid.uuid4()),
                    event_type=StreamEventType.HEARTBEAT,
                    timestamp=datetime.now(),
                    data={
                        "pipeline_status": "running",
                        "buffer_stats": await self.buffer.get_stats(),
                        "detector_stats": await self.detector.get_detection_stats(),
                    },
                    source="pipeline_heartbeat",
                )

                await self.event_processor.process_event(heartbeat_event)

                # Wait for next heartbeat
                await asyncio.sleep(self.config.heartbeat_interval_seconds)

        except Exception as e:
            logger.error(f"Heartbeat loop error: {e}")
        finally:
            logger.info("Heartbeat loop stopped")

    async def get_pipeline_status(self) -> dict[str, Any]:
        """Get comprehensive pipeline status.

        Returns:
            Pipeline status information
        """
        return {
            "is_running": self.is_running,
            "config": self.config.to_dict(),
            "buffer_stats": await self.buffer.get_stats(),
            "detector_stats": await self.detector.get_detection_stats(),
            "processor_stats": await self.event_processor.get_processing_stats(),
            "timestamp": datetime.now().isoformat(),
        }


class StreamProcessor:
    """Main stream processor facade."""

    def __init__(self):
        """Initialize stream processor."""
        self.pipelines: dict[str, StreamingPipeline] = {}
        self.global_stats = {
            "pipelines_created": 0,
            "total_events_processed": 0,
            "start_time": datetime.now(),
        }

    async def create_pipeline(
        self,
        pipeline_id: str,
        detector: Detector,
        config: StreamingConfig | None = None,
    ) -> bool:
        """Create new streaming pipeline.

        Args:
            pipeline_id: Pipeline identifier
            detector: Anomaly detector
            config: Streaming configuration

        Returns:
            True if created successfully
        """
        try:
            if pipeline_id in self.pipelines:
                logger.warning(f"Pipeline {pipeline_id} already exists")
                return False

            pipeline = StreamingPipeline(detector, config)
            self.pipelines[pipeline_id] = pipeline
            self.global_stats["pipelines_created"] += 1

            logger.info(f"Pipeline {pipeline_id} created")
            return True

        except Exception as e:
            logger.error(f"Failed to create pipeline {pipeline_id}: {e}")
            return False

    async def start_pipeline(self, pipeline_id: str) -> bool:
        """Start specific pipeline.

        Args:
            pipeline_id: Pipeline identifier

        Returns:
            True if started successfully
        """
        if pipeline_id not in self.pipelines:
            logger.error(f"Pipeline {pipeline_id} not found")
            return False

        return await self.pipelines[pipeline_id].start()

    async def stop_pipeline(self, pipeline_id: str) -> bool:
        """Stop specific pipeline.

        Args:
            pipeline_id: Pipeline identifier

        Returns:
            True if stopped successfully
        """
        if pipeline_id not in self.pipelines:
            logger.error(f"Pipeline {pipeline_id} not found")
            return False

        return await self.pipelines[pipeline_id].stop()

    async def submit_data(
        self, pipeline_id: str, data_point: dict[str, Any], source: str = ""
    ) -> bool:
        """Submit data to specific pipeline.

        Args:
            pipeline_id: Pipeline identifier
            data_point: Data to process
            source: Data source

        Returns:
            True if submitted successfully
        """
        if pipeline_id not in self.pipelines:
            logger.error(f"Pipeline {pipeline_id} not found")
            return False

        success = await self.pipelines[pipeline_id].submit_data(data_point, source)
        if success:
            self.global_stats["total_events_processed"] += 1

        return success

    async def get_all_pipeline_status(self) -> dict[str, Any]:
        """Get status of all pipelines.

        Returns:
            Status information for all pipelines
        """
        pipeline_statuses = {}

        for pipeline_id, pipeline in self.pipelines.items():
            pipeline_statuses[pipeline_id] = await pipeline.get_pipeline_status()

        return {
            "global_stats": self.global_stats,
            "active_pipelines": len(
                [p for p in self.pipelines.values() if p.is_running]
            ),
            "total_pipelines": len(self.pipelines),
            "pipelines": pipeline_statuses,
            "timestamp": datetime.now().isoformat(),
        }

    async def shutdown_all(self) -> bool:
        """Shutdown all pipelines.

        Returns:
            True if all pipelines shut down successfully
        """
        results = []

        for pipeline_id, pipeline in self.pipelines.items():
            if pipeline.is_running:
                result = await pipeline.stop()
                results.append(result)
                logger.info(f"Pipeline {pipeline_id} shutdown: {result}")

        return all(results)


# Global stream processor
_stream_processor: StreamProcessor | None = None


def get_stream_processor() -> StreamProcessor:
    """Get global stream processor.

    Returns:
        Stream processor instance
    """
    global _stream_processor

    if _stream_processor is None:
        _stream_processor = StreamProcessor()

    return _stream_processor
