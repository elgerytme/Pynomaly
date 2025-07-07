"""
Streaming data processing service for real-time anomaly detection.

This module provides a complete streaming data processing pipeline for 
real-time anomaly detection with backpressure handling, windowing, 
and distributed processing capabilities.
"""

import asyncio
import json
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Union
from enum import Enum
import threading
from queue import Queue, Empty

import numpy as np
import pandas as pd

from pynomaly.domain.entities.anomaly import Anomaly, Score
from pynomaly.domain.entities.dataset import Dataset
from pynomaly.domain.entities.detector import Detector
from pynomaly.shared.exceptions import StreamingError, ProcessingError
from pynomaly.shared.types import TenantId, UserId

logger = logging.getLogger(__name__)


class StreamState(str, Enum):
    """Stream processing states."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    STOPPING = "stopping"


class WindowType(str, Enum):
    """Types of windowing for stream processing."""
    TUMBLING = "tumbling"  # Non-overlapping fixed-size windows
    SLIDING = "sliding"    # Overlapping windows
    SESSION = "session"    # Variable-size based on activity gaps


@dataclass
class StreamMetrics:
    """Metrics for stream processing performance."""
    total_processed: int = 0
    anomalies_detected: int = 0
    processing_rate: float = 0.0  # Records per second
    avg_latency_ms: float = 0.0
    error_count: int = 0
    last_processed: Optional[datetime] = None
    backpressure_events: int = 0
    dropped_records: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "total_processed": self.total_processed,
            "anomalies_detected": self.anomalies_detected,
            "processing_rate": self.processing_rate,
            "avg_latency_ms": self.avg_latency_ms,
            "error_count": self.error_count,
            "last_processed": self.last_processed.isoformat() if self.last_processed else None,
            "backpressure_events": self.backpressure_events,
            "dropped_records": self.dropped_records
        }


@dataclass
class StreamRecord:
    """Individual record in the stream."""
    id: str
    timestamp: datetime
    data: Dict[str, Any]
    tenant_id: TenantId
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert record to pandas DataFrame."""
        return pd.DataFrame([self.data])


@dataclass
class StreamWindow:
    """Window of streaming data."""
    window_id: str
    start_time: datetime
    end_time: datetime
    records: List[StreamRecord]
    window_type: WindowType
    
    @property
    def size(self) -> int:
        """Number of records in window."""
        return len(self.records)
    
    def to_dataset(self) -> Dataset:
        """Convert window to Dataset for processing."""
        if not self.records:
            return Dataset(
                id=self.window_id,
                data=pd.DataFrame(),
                metadata={"window_type": self.window_type.value}
            )
        
        # Combine all record data
        data_list = [record.data for record in self.records]
        df = pd.DataFrame(data_list)
        
        return Dataset(
            id=self.window_id,
            data=df,
            metadata={
                "window_type": self.window_type.value,
                "window_start": self.start_time.isoformat(),
                "window_end": self.end_time.isoformat(),
                "record_count": len(self.records)
            }
        )


class BackpressureHandler:
    """Handles backpressure in streaming processing."""
    
    def __init__(
        self,
        max_queue_size: int = 10000,
        drop_strategy: str = "drop_oldest",  # "drop_oldest", "drop_newest", "reject"
        pressure_threshold: float = 0.8
    ):
        self.max_queue_size = max_queue_size
        self.drop_strategy = drop_strategy
        self.pressure_threshold = pressure_threshold
        self.current_queue_size = 0
        
    def should_apply_backpressure(self) -> bool:
        """Check if backpressure should be applied."""
        pressure_ratio = self.current_queue_size / self.max_queue_size
        return pressure_ratio >= self.pressure_threshold
    
    def handle_overflow(self, queue: deque, new_record: StreamRecord) -> bool:
        """Handle queue overflow. Returns True if record was accepted."""
        if self.drop_strategy == "drop_oldest":
            if queue:
                queue.popleft()
            queue.append(new_record)
            return True
        elif self.drop_strategy == "drop_newest":
            # Don't add new record
            return False
        elif self.drop_strategy == "reject":
            return False
        
        return False


class StreamingProcessor:
    """Main streaming processor for real-time anomaly detection."""
    
    def __init__(
        self,
        detector: Detector,
        window_config: Dict[str, Any],
        backpressure_config: Optional[Dict[str, Any]] = None,
        buffer_size: int = 1000,
        max_batch_size: int = 100,
        processing_timeout: float = 30.0
    ):
        self.detector = detector
        self.window_config = window_config
        self.buffer_size = buffer_size
        self.max_batch_size = max_batch_size
        self.processing_timeout = processing_timeout
        
        # Initialize backpressure handler
        backpressure_config = backpressure_config or {}
        self.backpressure_handler = BackpressureHandler(**backpressure_config)
        
        # Stream state
        self.state = StreamState.STOPPED
        self.record_buffer = deque(maxlen=buffer_size)
        self.windows = {}  # window_id -> StreamWindow
        self.metrics = StreamMetrics()
        
        # Processing components
        self._processing_task = None
        self._window_manager_task = None
        self._shutdown_event = asyncio.Event()
        
        # Callbacks
        self.anomaly_callbacks: List[Callable[[Anomaly], None]] = []
        self.metrics_callbacks: List[Callable[[StreamMetrics], None]] = []
        
        logger.info(f"Initialized StreamingProcessor with detector: {detector.name}")
    
    async def start(self) -> None:
        """Start the streaming processor."""
        if self.state != StreamState.STOPPED:
            raise StreamingError(f"Cannot start processor in state: {self.state}")
        
        logger.info("Starting streaming processor...")
        self.state = StreamState.STARTING
        
        try:
            # Start processing tasks
            self._processing_task = asyncio.create_task(self._processing_loop())
            self._window_manager_task = asyncio.create_task(self._window_manager_loop())
            
            self.state = StreamState.RUNNING
            logger.info("Streaming processor started successfully")
            
        except Exception as e:
            self.state = StreamState.ERROR
            logger.error(f"Failed to start streaming processor: {e}")
            raise StreamingError(f"Failed to start processor: {e}")
    
    async def stop(self) -> None:
        """Stop the streaming processor."""
        if self.state == StreamState.STOPPED:
            return
        
        logger.info("Stopping streaming processor...")
        self.state = StreamState.STOPPING
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Wait for tasks to complete
        if self._processing_task:
            try:
                await asyncio.wait_for(self._processing_task, timeout=10.0)
            except asyncio.TimeoutError:
                logger.warning("Processing task did not stop gracefully")
                self._processing_task.cancel()
        
        if self._window_manager_task:
            try:
                await asyncio.wait_for(self._window_manager_task, timeout=10.0)
            except asyncio.TimeoutError:
                logger.warning("Window manager task did not stop gracefully")
                self._window_manager_task.cancel()
        
        self.state = StreamState.STOPPED
        logger.info("Streaming processor stopped")
    
    async def process_record(self, record: StreamRecord) -> bool:
        """Process a single streaming record. Returns True if accepted."""
        if self.state != StreamState.RUNNING:
            return False
        
        # Check backpressure
        self.backpressure_handler.current_queue_size = len(self.record_buffer)
        
        if len(self.record_buffer) >= self.buffer_size:
            # Handle overflow
            accepted = self.backpressure_handler.handle_overflow(self.record_buffer, record)
            if not accepted:
                self.metrics.dropped_records += 1
                self.metrics.backpressure_events += 1
                return False
            else:
                self.metrics.backpressure_events += 1
        else:
            self.record_buffer.append(record)
        
        self.metrics.total_processed += 1
        self.metrics.last_processed = datetime.utcnow()
        
        return True
    
    async def process_batch(self, records: List[StreamRecord]) -> int:
        """Process a batch of records. Returns number of accepted records."""
        accepted_count = 0
        
        for record in records:
            if await self.process_record(record):
                accepted_count += 1
        
        return accepted_count
    
    def add_anomaly_callback(self, callback: Callable[[Anomaly], None]) -> None:
        """Add callback for anomaly detection events."""
        self.anomaly_callbacks.append(callback)
    
    def add_metrics_callback(self, callback: Callable[[StreamMetrics], None]) -> None:
        """Add callback for metrics updates."""
        self.metrics_callbacks.append(callback)
    
    def get_metrics(self) -> StreamMetrics:
        """Get current streaming metrics."""
        return self.metrics
    
    def get_state(self) -> StreamState:
        """Get current processor state."""
        return self.state
    
    async def _processing_loop(self) -> None:
        """Main processing loop."""
        logger.info("Starting processing loop")
        
        while not self._shutdown_event.is_set():
            try:
                # Process windows that are ready
                ready_windows = await self._get_ready_windows()
                
                for window in ready_windows:
                    await self._process_window(window)
                
                # Update metrics
                await self._update_metrics()
                
                # Small delay to prevent tight loop
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                self.metrics.error_count += 1
                
                # Pause briefly on error
                await asyncio.sleep(1.0)
    
    async def _window_manager_loop(self) -> None:
        """Window management loop."""
        logger.info("Starting window manager loop")
        
        while not self._shutdown_event.is_set():
            try:
                # Create windows from buffered records
                await self._create_windows()
                
                # Clean up old windows
                await self._cleanup_old_windows()
                
                # Window management runs less frequently
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Error in window manager loop: {e}")
                await asyncio.sleep(1.0)
    
    async def _create_windows(self) -> None:
        """Create windows from buffered records."""
        if not self.record_buffer:
            return
        
        window_type = WindowType(self.window_config.get("type", "tumbling"))
        window_size = timedelta(seconds=self.window_config.get("size_seconds", 60))
        
        if window_type == WindowType.TUMBLING:
            await self._create_tumbling_windows(window_size)
        elif window_type == WindowType.SLIDING:
            slide_interval = timedelta(seconds=self.window_config.get("slide_seconds", 30))
            await self._create_sliding_windows(window_size, slide_interval)
        elif window_type == WindowType.SESSION:
            gap_timeout = timedelta(seconds=self.window_config.get("gap_timeout_seconds", 300))
            await self._create_session_windows(gap_timeout)
    
    async def _create_tumbling_windows(self, window_size: timedelta) -> None:
        """Create non-overlapping tumbling windows."""
        records_to_process = []
        
        # Collect records for processing
        while self.record_buffer:
            records_to_process.append(self.record_buffer.popleft())
        
        if not records_to_process:
            return
        
        # Group records by time windows
        window_groups = {}
        
        for record in records_to_process:
            # Calculate which window this record belongs to
            window_start = self._align_to_window_boundary(record.timestamp, window_size)
            window_end = window_start + window_size
            window_key = window_start.isoformat()
            
            if window_key not in window_groups:
                window_groups[window_key] = {
                    "start": window_start,
                    "end": window_end,
                    "records": []
                }
            
            window_groups[window_key]["records"].append(record)
        
        # Create windows
        for window_key, group in window_groups.items():
            window_id = f"tumbling_{window_key}_{uuid.uuid4().hex[:8]}"
            
            window = StreamWindow(
                window_id=window_id,
                start_time=group["start"],
                end_time=group["end"],
                records=group["records"],
                window_type=WindowType.TUMBLING
            )
            
            self.windows[window_id] = window
    
    async def _create_sliding_windows(self, window_size: timedelta, slide_interval: timedelta) -> None:
        """Create overlapping sliding windows."""
        # Simplified sliding window implementation
        # In a full implementation, you'd maintain overlapping windows
        await self._create_tumbling_windows(slide_interval)
    
    async def _create_session_windows(self, gap_timeout: timedelta) -> None:
        """Create session-based windows with gap detection."""
        records_to_process = []
        
        while self.record_buffer:
            records_to_process.append(self.record_buffer.popleft())
        
        if not records_to_process:
            return
        
        # Sort records by timestamp
        records_to_process.sort(key=lambda r: r.timestamp)
        
        # Group into sessions based on gaps
        current_session = []
        sessions = []
        
        for record in records_to_process:
            if current_session:
                # Check if gap is too large
                time_gap = record.timestamp - current_session[-1].timestamp
                if time_gap > gap_timeout:
                    # Start new session
                    sessions.append(current_session)
                    current_session = [record]
                else:
                    current_session.append(record)
            else:
                current_session.append(record)
        
        # Add final session
        if current_session:
            sessions.append(current_session)
        
        # Create windows from sessions
        for i, session_records in enumerate(sessions):
            if not session_records:
                continue
            
            window_id = f"session_{datetime.utcnow().isoformat()}_{i}"
            start_time = session_records[0].timestamp
            end_time = session_records[-1].timestamp
            
            window = StreamWindow(
                window_id=window_id,
                start_time=start_time,
                end_time=end_time,
                records=session_records,
                window_type=WindowType.SESSION
            )
            
            self.windows[window_id] = window
    
    def _align_to_window_boundary(self, timestamp: datetime, window_size: timedelta) -> datetime:
        """Align timestamp to window boundary."""
        # Simple alignment to window boundaries
        epoch = datetime(1970, 1, 1)
        seconds_since_epoch = (timestamp - epoch).total_seconds()
        window_seconds = window_size.total_seconds()
        
        aligned_seconds = int(seconds_since_epoch // window_seconds) * window_seconds
        return epoch + timedelta(seconds=aligned_seconds)
    
    async def _get_ready_windows(self) -> List[StreamWindow]:
        """Get windows that are ready for processing."""
        ready_windows = []
        current_time = datetime.utcnow()
        
        # Check if windows are complete
        for window_id, window in list(self.windows.items()):
            # Window is ready if current time has passed the window end time
            if current_time >= window.end_time:
                ready_windows.append(window)
                # Remove from pending windows
                del self.windows[window_id]
        
        return ready_windows
    
    async def _process_window(self, window: StreamWindow) -> None:
        """Process a complete window."""
        start_time = time.time()
        
        try:
            logger.debug(f"Processing window {window.window_id} with {window.size} records")
            
            # Convert window to dataset
            dataset = window.to_dataset()
            
            if dataset.data.empty:
                logger.debug(f"Skipping empty window {window.window_id}")
                return
            
            # Perform anomaly detection
            result = await self._detect_anomalies(dataset)
            
            # Process detected anomalies
            for anomaly in result.anomalies:
                await self._handle_anomaly(anomaly)
            
            # Update metrics
            self.metrics.anomalies_detected += len(result.anomalies)
            
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            self._update_latency_metric(processing_time)
            
            logger.debug(f"Processed window {window.window_id}: {len(result.anomalies)} anomalies detected")
            
        except Exception as e:
            logger.error(f"Error processing window {window.window_id}: {e}")
            self.metrics.error_count += 1
            raise ProcessingError(f"Window processing failed: {e}")
    
    async def _detect_anomalies(self, dataset: Dataset):
        """Perform anomaly detection on dataset."""
        # Use the configured detector
        if not self.detector.is_trained:
            raise ProcessingError("Detector is not trained")
        
        # This would call the actual detector's predict method
        # For now, we'll simulate the detection
        from pynomaly.domain.entities.detection_result import DetectionResult
        
        # In a real implementation, this would call:
        # result = await self.detector.detect(dataset)
        
        # Simulated detection for demonstration
        anomalies = []
        if len(dataset.data) > 0:
            # Simple simulation - mark 5% as anomalies randomly
            n_anomalies = max(1, int(len(dataset.data) * 0.05))
            anomaly_indices = np.random.choice(len(dataset.data), n_anomalies, replace=False)
            
            for idx in anomaly_indices:
                anomaly = Anomaly(
                    index=int(idx),
                    score=Score(float(np.random.random())),
                    features=dataset.data.iloc[idx].to_dict(),
                    metadata={
                        "detector": self.detector.name,
                        "window_id": dataset.id,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
                anomalies.append(anomaly)
        
        return DetectionResult(
            anomalies=anomalies,
            algorithm=self.detector.algorithm,
            threshold=0.5,
            metadata={
                "window_size": len(dataset.data),
                "processing_time": datetime.utcnow().isoformat()
            }
        )
    
    async def _handle_anomaly(self, anomaly: Anomaly) -> None:
        """Handle detected anomaly."""
        # Call registered callbacks
        for callback in self.anomaly_callbacks:
            try:
                callback(anomaly)
            except Exception as e:
                logger.error(f"Error in anomaly callback: {e}")
    
    async def _update_metrics(self) -> None:
        """Update processing metrics."""
        # Calculate processing rate
        if self.metrics.last_processed:
            time_diff = (datetime.utcnow() - self.metrics.last_processed).total_seconds()
            if time_diff > 0:
                # Exponential moving average for processing rate
                current_rate = 1.0 / time_diff if time_diff > 0 else 0.0
                alpha = 0.1  # Smoothing factor
                self.metrics.processing_rate = (alpha * current_rate + 
                                              (1 - alpha) * self.metrics.processing_rate)
        
        # Call metrics callbacks
        for callback in self.metrics_callbacks:
            try:
                callback(self.metrics)
            except Exception as e:
                logger.error(f"Error in metrics callback: {e}")
    
    def _update_latency_metric(self, processing_time_ms: float) -> None:
        """Update average latency metric."""
        # Exponential moving average
        alpha = 0.1
        self.metrics.avg_latency_ms = (alpha * processing_time_ms + 
                                     (1 - alpha) * self.metrics.avg_latency_ms)
    
    async def _cleanup_old_windows(self) -> None:
        """Clean up old windows that are no longer needed."""
        current_time = datetime.utcnow()
        cleanup_threshold = current_time - timedelta(hours=1)  # Keep windows for 1 hour
        
        windows_to_remove = []
        for window_id, window in self.windows.items():
            if window.end_time < cleanup_threshold:
                windows_to_remove.append(window_id)
        
        for window_id in windows_to_remove:
            del self.windows[window_id]
            logger.debug(f"Cleaned up old window: {window_id}")


class StreamingService:
    """High-level service for managing multiple streaming processors."""
    
    def __init__(self):
        self.processors: Dict[str, StreamingProcessor] = {}
        self.tenant_configs: Dict[TenantId, Dict[str, Any]] = {}
        
    async def create_processor(
        self,
        processor_id: str,
        detector: Detector,
        config: Dict[str, Any]
    ) -> StreamingProcessor:
        """Create a new streaming processor."""
        if processor_id in self.processors:
            raise StreamingError(f"Processor {processor_id} already exists")
        
        processor = StreamingProcessor(
            detector=detector,
            window_config=config.get("window", {}),
            backpressure_config=config.get("backpressure", {}),
            buffer_size=config.get("buffer_size", 1000),
            max_batch_size=config.get("max_batch_size", 100),
            processing_timeout=config.get("processing_timeout", 30.0)
        )
        
        self.processors[processor_id] = processor
        logger.info(f"Created streaming processor: {processor_id}")
        
        return processor
    
    async def start_processor(self, processor_id: str) -> None:
        """Start a streaming processor."""
        if processor_id not in self.processors:
            raise StreamingError(f"Processor {processor_id} not found")
        
        await self.processors[processor_id].start()
    
    async def stop_processor(self, processor_id: str) -> None:
        """Stop a streaming processor."""
        if processor_id not in self.processors:
            raise StreamingError(f"Processor {processor_id} not found")
        
        await self.processors[processor_id].stop()
    
    async def remove_processor(self, processor_id: str) -> None:
        """Remove a streaming processor."""
        if processor_id in self.processors:
            await self.stop_processor(processor_id)
            del self.processors[processor_id]
            logger.info(f"Removed streaming processor: {processor_id}")
    
    def get_processor(self, processor_id: str) -> Optional[StreamingProcessor]:
        """Get a streaming processor by ID."""
        return self.processors.get(processor_id)
    
    def list_processors(self) -> List[str]:
        """List all processor IDs."""
        return list(self.processors.keys())
    
    async def get_all_metrics(self) -> Dict[str, StreamMetrics]:
        """Get metrics for all processors."""
        metrics = {}
        for processor_id, processor in self.processors.items():
            metrics[processor_id] = processor.get_metrics()
        return metrics
    
    async def shutdown(self) -> None:
        """Shutdown all processors."""
        logger.info("Shutting down streaming service...")
        
        # Stop all processors
        for processor_id in list(self.processors.keys()):
            await self.remove_processor(processor_id)
        
        logger.info("Streaming service shutdown complete")