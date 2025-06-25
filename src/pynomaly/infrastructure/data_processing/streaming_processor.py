"""Streaming data processor for real-time anomaly detection.

This module provides streaming capabilities for processing data in real-time
or near-real-time scenarios, supporting both batch and continuous streaming modes.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Protocol
from uuid import uuid4

import pandas as pd

from pynomaly.domain.entities import DetectionResult
from pynomaly.domain.exceptions import InfrastructureError
from pynomaly.infrastructure.monitoring import get_monitor, monitor_async_operation

from .memory_efficient_processor import get_memory_usage


class StreamingMode(Enum):
    """Streaming processing modes."""

    BATCH = "batch"
    CONTINUOUS = "continuous"
    WINDOW = "window"
    EVENT_DRIVEN = "event_driven"


class BackpressureStrategy(Enum):
    """Backpressure handling strategies."""

    BLOCK = "block"
    DROP_OLDEST = "drop_oldest"
    DROP_NEWEST = "drop_newest"
    SAMPLE = "sample"


@dataclass
class StreamingConfig:
    """Configuration for streaming processor."""

    mode: StreamingMode = StreamingMode.BATCH
    batch_size: int = 1000
    window_size: timedelta = timedelta(minutes=5)
    max_queue_size: int = 10000
    backpressure_strategy: BackpressureStrategy = BackpressureStrategy.BLOCK
    processing_timeout: float = 30.0
    enable_metrics: bool = True
    enable_checkpointing: bool = False
    checkpoint_interval: int = 1000


@dataclass
class StreamingMetrics:
    """Metrics for streaming operations."""

    messages_processed: int = 0
    messages_failed: int = 0
    average_latency_ms: float = 0.0
    throughput_per_second: float = 0.0
    queue_size: int = 0
    backpressure_events: int = 0
    last_update: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "messages_processed": self.messages_processed,
            "messages_failed": self.messages_failed,
            "average_latency_ms": self.average_latency_ms,
            "throughput_per_second": self.throughput_per_second,
            "queue_size": self.queue_size,
            "backpressure_events": self.backpressure_events,
            "last_update": self.last_update.isoformat(),
        }


@dataclass
class StreamingMessage:
    """Message container for streaming data."""

    message_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    data: pd.DataFrame | dict[str, Any] | list[dict[str, Any]] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    partition_key: str | None = None

    @property
    def size_bytes(self) -> int:
        """Estimate message size in bytes."""
        if isinstance(self.data, pd.DataFrame):
            return self.data.memory_usage(deep=True).sum()
        elif isinstance(self.data, (dict, list)):
            return len(str(self.data).encode("utf-8"))
        return 0


class StreamingDataSource(Protocol):
    """Protocol for streaming data sources."""

    async def read_messages(self, max_messages: int = 100) -> list[StreamingMessage]:
        """Read messages from the data source."""
        ...

    async def acknowledge(self, message_ids: list[str]) -> None:
        """Acknowledge processed messages."""
        ...


class StreamingDataSink(Protocol):
    """Protocol for streaming data sinks."""

    async def write_results(self, results: list[DetectionResult]) -> None:
        """Write detection results to the sink."""
        ...


class StreamingAnomalyDetector(ABC):
    """Abstract base class for streaming anomaly detectors."""

    @abstractmethod
    async def detect_streaming(
        self, message: StreamingMessage
    ) -> DetectionResult | None:
        """Detect anomalies in streaming message."""
        ...

    @abstractmethod
    async def update_model(self, message: StreamingMessage) -> None:
        """Update model with new data (for online learning)."""
        ...


class StreamingProcessor:
    """Main streaming processor for anomaly detection."""

    def __init__(
        self,
        config: StreamingConfig | None = None,
        detector: StreamingAnomalyDetector | None = None,
    ):
        """Initialize streaming processor.

        Args:
            config: Streaming configuration
            detector: Anomaly detector for streaming
        """
        self.config = config or StreamingConfig()
        self.detector = detector

        # Internal state
        self._message_queue: deque = deque(maxlen=self.config.max_queue_size)
        self._processing_windows: dict[str, deque] = {}
        self._metrics = StreamingMetrics()
        self._running = False
        self._tasks: list[asyncio.Task] = []

        # Checkpointing
        self._checkpoint_counter = 0
        self._last_checkpoint: datetime | None = None

        # Latency tracking
        self._latency_samples: deque = deque(maxlen=1000)

    async def start(
        self, source: StreamingDataSource, sink: StreamingDataSink | None = None
    ) -> None:
        """Start streaming processing.

        Args:
            source: Data source to read from
            sink: Data sink to write results to
        """
        if self._running:
            raise InfrastructureError("Streaming processor already running")

        if not self.detector:
            raise InfrastructureError("No detector configured for streaming")

        self._running = True

        async with monitor_async_operation(
            "streaming_processing", "streaming_processor"
        ):
            get_monitor().info(
                "Starting streaming processor",
                operation="streaming_start",
                component="streaming_processor",
                mode=self.config.mode.value,
                batch_size=self.config.batch_size,
            )

            try:
                # Start processing tasks
                if self.config.mode == StreamingMode.CONTINUOUS:
                    await self._process_continuous(source, sink)
                elif self.config.mode == StreamingMode.BATCH:
                    await self._process_batch(source, sink)
                elif self.config.mode == StreamingMode.WINDOW:
                    await self._process_window(source, sink)
                else:
                    raise InfrastructureError(
                        f"Unsupported streaming mode: {self.config.mode}"
                    )

            except Exception as e:
                get_monitor().error(
                    f"Streaming processing error: {e}",
                    operation="streaming_processing",
                    component="streaming_processor",
                    error_type=type(e).__name__,
                )
                raise
            finally:
                await self.stop()

    async def stop(self) -> None:
        """Stop streaming processing."""
        if not self._running:
            return

        self._running = False

        # Cancel all tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()

        # Wait for tasks to complete
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        get_monitor().info(
            "Streaming processor stopped",
            operation="streaming_stop",
            component="streaming_processor",
            final_metrics=self._metrics.to_dict(),
        )

    async def _process_continuous(
        self, source: StreamingDataSource, sink: StreamingDataSink | None
    ) -> None:
        """Process messages continuously."""
        while self._running:
            try:
                # Read messages from source
                messages = await source.read_messages(
                    max_messages=self.config.batch_size
                )

                if not messages:
                    await asyncio.sleep(0.1)  # Brief pause if no messages
                    continue

                # Process messages
                results = []
                processed_message_ids = []

                for message in messages:
                    start_time = time.time()

                    try:
                        # Detect anomalies
                        result = await self.detector.detect_streaming(message)
                        if result:
                            results.append(result)

                        # Update model if online learning is enabled
                        await self.detector.update_model(message)

                        processed_message_ids.append(message.message_id)
                        self._metrics.messages_processed += 1

                        # Track latency
                        latency_ms = (time.time() - start_time) * 1000
                        self._latency_samples.append(latency_ms)

                    except Exception as e:
                        self._metrics.messages_failed += 1
                        get_monitor().error(
                            f"Failed to process message {message.message_id}: {e}",
                            operation="message_processing",
                            component="streaming_processor",
                            message_id=message.message_id,
                        )

                # Write results to sink
                if results and sink:
                    await sink.write_results(results)

                # Acknowledge processed messages
                if processed_message_ids:
                    await source.acknowledge(processed_message_ids)

                # Update metrics
                self._update_metrics()

                # Checkpointing
                if self.config.enable_checkpointing:
                    await self._maybe_checkpoint()

            except Exception as e:
                get_monitor().error(
                    f"Error in continuous processing loop: {e}",
                    operation="continuous_processing",
                    component="streaming_processor",
                )
                await asyncio.sleep(1.0)  # Back off on error

    async def _process_batch(
        self, source: StreamingDataSource, sink: StreamingDataSink | None
    ) -> None:
        """Process messages in batches."""
        while self._running:
            try:
                # Collect batch of messages
                batch_messages = []
                for _ in range(self.config.batch_size):
                    messages = await source.read_messages(max_messages=1)
                    if messages:
                        batch_messages.extend(messages)
                    else:
                        break

                if not batch_messages:
                    await asyncio.sleep(0.1)
                    continue

                # Process batch
                batch_results = []
                processed_message_ids = []

                start_time = time.time()

                for message in batch_messages:
                    try:
                        result = await self.detector.detect_streaming(message)
                        if result:
                            batch_results.append(result)

                        await self.detector.update_model(message)
                        processed_message_ids.append(message.message_id)
                        self._metrics.messages_processed += 1

                    except Exception as e:
                        self._metrics.messages_failed += 1
                        get_monitor().error(
                            f"Failed to process message in batch: {e}",
                            operation="batch_processing",
                            component="streaming_processor",
                            message_id=message.message_id,
                        )

                # Track batch latency
                batch_latency_ms = (time.time() - start_time) * 1000
                avg_latency = (
                    batch_latency_ms / len(batch_messages) if batch_messages else 0
                )
                self._latency_samples.append(avg_latency)

                # Write batch results
                if batch_results and sink:
                    await sink.write_results(batch_results)

                # Acknowledge batch
                if processed_message_ids:
                    await source.acknowledge(processed_message_ids)

                self._update_metrics()

                if self.config.enable_checkpointing:
                    await self._maybe_checkpoint()

            except Exception as e:
                get_monitor().error(
                    f"Error in batch processing: {e}",
                    operation="batch_processing",
                    component="streaming_processor",
                )
                await asyncio.sleep(1.0)

    async def _process_window(
        self, source: StreamingDataSource, sink: StreamingDataSink | None
    ) -> None:
        """Process messages using time windows."""
        window_start = datetime.utcnow()
        window_messages: list[StreamingMessage] = []

        while self._running:
            try:
                current_time = datetime.utcnow()

                # Check if window has expired
                if current_time - window_start >= self.config.window_size:
                    if window_messages:
                        # Process window
                        await self._process_window_batch(window_messages, sink, source)

                    # Start new window
                    window_start = current_time
                    window_messages = []

                # Read new messages
                messages = await source.read_messages(
                    max_messages=self.config.batch_size
                )
                window_messages.extend(messages)

                # Handle backpressure
                if len(window_messages) > self.config.max_queue_size:
                    await self._handle_backpressure(window_messages)

                await asyncio.sleep(0.01)  # Small delay to prevent tight loop

            except Exception as e:
                get_monitor().error(
                    f"Error in window processing: {e}",
                    operation="window_processing",
                    component="streaming_processor",
                )
                await asyncio.sleep(1.0)

    async def _process_window_batch(
        self,
        messages: list[StreamingMessage],
        sink: StreamingDataSink | None,
        source: StreamingDataSource,
    ) -> None:
        """Process a window batch of messages."""
        start_time = time.time()
        results = []
        processed_message_ids = []

        # Combine messages into single dataset if possible
        combined_data = self._combine_messages(messages)

        # Process combined data
        for message in messages:
            try:
                result = await self.detector.detect_streaming(message)
                if result:
                    results.append(result)

                await self.detector.update_model(message)
                processed_message_ids.append(message.message_id)
                self._metrics.messages_processed += 1

            except Exception as e:
                self._metrics.messages_failed += 1
                get_monitor().error(
                    f"Failed to process message in window: {e}",
                    operation="window_processing",
                    component="streaming_processor",
                    message_id=message.message_id,
                )

        # Track window latency
        window_latency_ms = (time.time() - start_time) * 1000
        avg_latency = window_latency_ms / len(messages) if messages else 0
        self._latency_samples.append(avg_latency)

        # Write results
        if results and sink:
            await sink.write_results(results)

        # Acknowledge messages
        if processed_message_ids:
            await source.acknowledge(processed_message_ids)

        get_monitor().info(
            f"Processed window with {len(messages)} messages",
            operation="window_processing",
            component="streaming_processor",
            window_size=len(messages),
            results_count=len(results),
            latency_ms=window_latency_ms,
        )

    def _combine_messages(
        self, messages: list[StreamingMessage]
    ) -> pd.DataFrame | None:
        """Combine multiple messages into a single DataFrame."""
        try:
            dataframes = []

            for message in messages:
                if isinstance(message.data, pd.DataFrame):
                    dataframes.append(message.data)
                elif isinstance(message.data, dict):
                    dataframes.append(pd.DataFrame([message.data]))
                elif isinstance(message.data, list):
                    dataframes.append(pd.DataFrame(message.data))

            if dataframes:
                return pd.concat(dataframes, ignore_index=True)

        except Exception as e:
            get_monitor().warning(
                f"Could not combine messages: {e}",
                operation="message_combination",
                component="streaming_processor",
            )

        return None

    async def _handle_backpressure(self, messages: list[StreamingMessage]) -> None:
        """Handle backpressure when queue is full."""
        self._metrics.backpressure_events += 1

        if self.config.backpressure_strategy == BackpressureStrategy.DROP_OLDEST:
            # Remove oldest messages
            excess = len(messages) - self.config.max_queue_size
            del messages[:excess]

        elif self.config.backpressure_strategy == BackpressureStrategy.DROP_NEWEST:
            # Keep only max_queue_size messages
            messages[:] = messages[: self.config.max_queue_size]

        elif self.config.backpressure_strategy == BackpressureStrategy.SAMPLE:
            # Randomly sample messages
            import random

            if len(messages) > self.config.max_queue_size:
                sampled = random.sample(messages, self.config.max_queue_size)
                messages[:] = sampled

        # BLOCK strategy doesn't modify messages - just logs the event
        get_monitor().warning(
            f"Backpressure event: {len(messages)} messages, strategy: {self.config.backpressure_strategy.value}",
            operation="backpressure_handling",
            component="streaming_processor",
            queue_size=len(messages),
            strategy=self.config.backpressure_strategy.value,
        )

    def _update_metrics(self) -> None:
        """Update streaming metrics."""
        now = datetime.utcnow()

        # Calculate throughput
        time_delta = (now - self._metrics.last_update).total_seconds()
        if time_delta > 0:
            self._metrics.throughput_per_second = (
                self._metrics.messages_processed / time_delta
            )

        # Calculate average latency
        if self._latency_samples:
            self._metrics.average_latency_ms = sum(self._latency_samples) / len(
                self._latency_samples
            )

        self._metrics.queue_size = len(self._message_queue)
        self._metrics.last_update = now

    async def _maybe_checkpoint(self) -> None:
        """Create checkpoint if needed."""
        self._checkpoint_counter += 1

        if self._checkpoint_counter >= self.config.checkpoint_interval:
            await self._create_checkpoint()
            self._checkpoint_counter = 0

    async def _create_checkpoint(self) -> None:
        """Create processing checkpoint."""
        checkpoint_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": self._metrics.to_dict(),
            "config": {
                "mode": self.config.mode.value,
                "batch_size": self.config.batch_size,
                "window_size": self.config.window_size.total_seconds(),
            },
        }

        self._last_checkpoint = datetime.utcnow()

        get_monitor().info(
            "Checkpoint created",
            operation="checkpointing",
            component="streaming_processor",
            checkpoint_data=checkpoint_data,
        )

    def get_metrics(self) -> StreamingMetrics:
        """Get current streaming metrics."""
        self._update_metrics()
        return self._metrics

    def get_status(self) -> dict[str, Any]:
        """Get processor status."""
        return {
            "running": self._running,
            "mode": self.config.mode.value,
            "metrics": self._metrics.to_dict(),
            "queue_size": len(self._message_queue),
            "last_checkpoint": self._last_checkpoint.isoformat()
            if self._last_checkpoint
            else None,
            "memory_usage": get_memory_usage().to_dict(),
        }
