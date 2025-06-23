"""Domain service for streaming anomaly detection."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import AsyncIterator, Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass
from collections import deque
import uuid

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class StreamingMode(Enum):
    """Streaming processing modes."""
    REAL_TIME = "real_time"
    BATCH = "batch"
    MICRO_BATCH = "micro_batch"
    ADAPTIVE = "adaptive"


class WindowType(Enum):
    """Types of sliding windows."""
    TIME_BASED = "time_based"
    COUNT_BASED = "count_based"
    SESSION_BASED = "session_based"
    ADAPTIVE_SIZE = "adaptive_size"


@dataclass
class StreamRecord:
    """Represents a single record in the stream."""
    id: str
    timestamp: datetime
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class StreamBatch:
    """Represents a batch of stream records."""
    batch_id: str
    records: List[StreamRecord]
    window_start: datetime
    window_end: datetime
    size: int


@dataclass
class StreamingResult:
    """Result from streaming anomaly detection."""
    record_id: str
    timestamp: datetime
    anomaly_score: float
    is_anomaly: bool
    confidence: float
    explanation: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class WindowConfiguration:
    """Configuration for sliding window."""
    window_type: WindowType
    size: Union[int, float]  # Count or time in seconds
    step: Union[int, float]  # Step size for sliding
    overlap: float = 0.0  # Overlap percentage
    min_size: Optional[int] = None  # Minimum window size
    max_size: Optional[int] = None  # Maximum window size


class StreamProcessorProtocol(ABC):
    """Protocol for stream processors."""
    
    @abstractmethod
    async def process_record(self, record: StreamRecord) -> Optional[StreamingResult]:
        """Process a single record."""
        pass
    
    @abstractmethod
    async def process_batch(self, batch: StreamBatch) -> List[StreamingResult]:
        """Process a batch of records."""
        pass
    
    @abstractmethod
    async def update_model(self, records: List[StreamRecord]) -> None:
        """Update the model with new data (online learning)."""
        pass


class StreamWindowManager:
    """Manages sliding windows for streaming data."""
    
    def __init__(self, config: WindowConfiguration):
        """Initialize window manager.
        
        Args:
            config: Window configuration
        """
        self.config = config
        self._buffer = deque()
        self._last_window_time: Optional[datetime] = None
        self._window_counter = 0
    
    def add_record(self, record: StreamRecord) -> Optional[StreamBatch]:
        """Add a record and check if window is ready.
        
        Args:
            record: Stream record to add
            
        Returns:
            StreamBatch if window is complete, None otherwise
        """
        self._buffer.append(record)
        
        if self.config.window_type == WindowType.COUNT_BASED:
            return self._check_count_window()
        elif self.config.window_type == WindowType.TIME_BASED:
            return self._check_time_window(record.timestamp)
        elif self.config.window_type == WindowType.SESSION_BASED:
            return self._check_session_window(record.timestamp)
        elif self.config.window_type == WindowType.ADAPTIVE_SIZE:
            return self._check_adaptive_window()
        
        return None
    
    def _check_count_window(self) -> Optional[StreamBatch]:
        """Check if count-based window is ready."""
        if len(self._buffer) >= self.config.size:
            return self._create_batch()
        return None
    
    def _check_time_window(self, current_time: datetime) -> Optional[StreamBatch]:
        """Check if time-based window is ready."""
        if not self._buffer:
            return None
        
        oldest_time = self._buffer[0].timestamp
        time_diff = (current_time - oldest_time).total_seconds()
        
        if time_diff >= self.config.size:
            return self._create_batch()
        return None
    
    def _check_session_window(self, current_time: datetime) -> Optional[StreamBatch]:
        """Check if session-based window is ready."""
        if not self._buffer or len(self._buffer) < 2:
            return None
        
        # Check for session timeout
        last_record_time = self._buffer[-2].timestamp
        time_gap = (current_time - last_record_time).total_seconds()
        
        if time_gap > self.config.size:  # Session timeout
            return self._create_batch()
        return None
    
    def _check_adaptive_window(self) -> Optional[StreamBatch]:
        """Check if adaptive window is ready."""
        # Simple adaptive strategy based on data variance
        if len(self._buffer) < self.config.min_size:
            return None
        
        if len(self._buffer) >= self.config.max_size:
            return self._create_batch()
        
        # Check data variance to decide window size
        if len(self._buffer) >= 10:  # Minimum for variance calculation
            recent_data = [r.data for r in list(self._buffer)[-10:]]
            if self._should_close_adaptive_window(recent_data):
                return self._create_batch()
        
        return None
    
    def _should_close_adaptive_window(self, recent_data: List[Dict[str, Any]]) -> bool:
        """Determine if adaptive window should be closed."""
        # Simple heuristic: close window if variance is low
        try:
            # Convert to numeric features for variance calculation
            numeric_values = []
            for record_data in recent_data:
                values = [v for v in record_data.values() if isinstance(v, (int, float))]
                if values:
                    numeric_values.extend(values)
            
            if len(numeric_values) > 1:
                variance = np.var(numeric_values)
                return variance < 0.1  # Threshold for low variance
        except Exception:
            pass
        
        return False
    
    def _create_batch(self) -> StreamBatch:
        """Create a batch from current buffer."""
        records = list(self._buffer)
        self._buffer.clear()
        
        self._window_counter += 1
        
        return StreamBatch(
            batch_id=f"batch_{self._window_counter}_{uuid.uuid4().hex[:8]}",
            records=records,
            window_start=records[0].timestamp if records else datetime.now(),
            window_end=records[-1].timestamp if records else datetime.now(),
            size=len(records)
        )
    
    def force_batch(self) -> Optional[StreamBatch]:
        """Force creation of batch from current buffer."""
        if self._buffer:
            return self._create_batch()
        return None


class StreamingDetectionService:
    """Domain service for streaming anomaly detection."""
    
    def __init__(
        self,
        window_config: WindowConfiguration,
        mode: StreamingMode = StreamingMode.REAL_TIME,
        buffer_size: int = 1000,
        anomaly_threshold: float = 0.5,
        enable_online_learning: bool = False
    ):
        """Initialize streaming detection service.
        
        Args:
            window_config: Window configuration
            mode: Streaming mode
            buffer_size: Maximum buffer size
            anomaly_threshold: Threshold for anomaly detection
            enable_online_learning: Whether to enable online learning
        """
        self.window_config = window_config
        self.mode = mode
        self.buffer_size = buffer_size
        self.anomaly_threshold = anomaly_threshold
        self.enable_online_learning = enable_online_learning
        
        self.window_manager = StreamWindowManager(window_config)
        self._processors: List[StreamProcessorProtocol] = []
        self._callbacks: List[Callable[[StreamingResult], None]] = []
        self._stats = {
            "processed_records": 0,
            "detected_anomalies": 0,
            "processing_errors": 0,
            "last_processing_time": None
        }
        self._running = False
    
    def register_processor(self, processor: StreamProcessorProtocol) -> None:
        """Register a stream processor."""
        self._processors.append(processor)
        logger.info(f"Registered stream processor: {processor.__class__.__name__}")
    
    def register_callback(self, callback: Callable[[StreamingResult], None]) -> None:
        """Register a callback for anomaly results."""
        self._callbacks.append(callback)
        logger.info("Registered anomaly detection callback")
    
    async def process_stream(
        self,
        stream: AsyncIterator[StreamRecord]
    ) -> AsyncIterator[StreamingResult]:
        """Process a stream of records.
        
        Args:
            stream: Async iterator of stream records
            
        Yields:
            StreamingResult for each processed record/batch
        """
        self._running = True
        
        try:
            async for record in stream:
                if not self._running:
                    break
                
                results = await self._process_record(record)
                for result in results:
                    yield result
                    # Execute callbacks
                    for callback in self._callbacks:
                        try:
                            callback(result)
                        except Exception as e:
                            logger.error(f"Callback execution failed: {e}")
        
        except Exception as e:
            logger.error(f"Stream processing error: {e}")
            self._stats["processing_errors"] += 1
            raise
        finally:
            self._running = False
    
    async def _process_record(self, record: StreamRecord) -> List[StreamingResult]:
        """Process a single record."""
        results = []
        
        try:
            # Update statistics
            self._stats["processed_records"] += 1
            self._stats["last_processing_time"] = datetime.now()
            
            if self.mode == StreamingMode.REAL_TIME:
                # Process record immediately
                for processor in self._processors:
                    result = await processor.process_record(record)
                    if result:
                        if result.is_anomaly:
                            self._stats["detected_anomalies"] += 1
                        results.append(result)
            
            elif self.mode in [StreamingMode.BATCH, StreamingMode.MICRO_BATCH]:
                # Add to window and check for batch processing
                batch = self.window_manager.add_record(record)
                if batch:
                    batch_results = await self._process_batch(batch)
                    results.extend(batch_results)
            
            elif self.mode == StreamingMode.ADAPTIVE:
                # Adaptive processing based on system load
                if await self._should_batch_process():
                    batch = self.window_manager.add_record(record)
                    if batch:
                        batch_results = await self._process_batch(batch)
                        results.extend(batch_results)
                else:
                    # Process in real-time
                    for processor in self._processors:
                        result = await processor.process_record(record)
                        if result:
                            if result.is_anomaly:
                                self._stats["detected_anomalies"] += 1
                            results.append(result)
        
        except Exception as e:
            logger.error(f"Record processing error: {e}")
            self._stats["processing_errors"] += 1
        
        return results
    
    async def _process_batch(self, batch: StreamBatch) -> List[StreamingResult]:
        """Process a batch of records."""
        results = []
        
        try:
            for processor in self._processors:
                batch_results = await processor.process_batch(batch)
                results.extend(batch_results)
                
                # Update online learning if enabled
                if self.enable_online_learning:
                    await processor.update_model(batch.records)
            
            # Update anomaly statistics
            anomaly_count = sum(1 for r in results if r.is_anomaly)
            self._stats["detected_anomalies"] += anomaly_count
            
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            self._stats["processing_errors"] += 1
        
        return results
    
    async def _should_batch_process(self) -> bool:
        """Determine if should use batch processing in adaptive mode."""
        # Simple heuristic based on processing load
        error_rate = (
            self._stats["processing_errors"] / max(self._stats["processed_records"], 1)
        )
        return error_rate > 0.1  # Switch to batch if error rate > 10%
    
    def stop(self) -> None:
        """Stop stream processing."""
        self._running = False
        logger.info("Stream processing stopped")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        stats = self._stats.copy()
        if stats["processed_records"] > 0:
            stats["anomaly_rate"] = stats["detected_anomalies"] / stats["processed_records"]
            stats["error_rate"] = stats["processing_errors"] / stats["processed_records"]
        else:
            stats["anomaly_rate"] = 0.0
            stats["error_rate"] = 0.0
        
        return stats
    
    async def flush_remaining(self) -> List[StreamingResult]:
        """Flush any remaining records in buffer."""
        results = []
        
        batch = self.window_manager.force_batch()
        if batch:
            batch_results = await self._process_batch(batch)
            results.extend(batch_results)
        
        return results


class StreamDataGenerator:
    """Utility class for generating synthetic streaming data."""
    
    def __init__(
        self,
        base_data: Optional[pd.DataFrame] = None,
        anomaly_rate: float = 0.05,
        noise_level: float = 0.1
    ):
        """Initialize data generator.
        
        Args:
            base_data: Base dataset to generate variations from
            anomaly_rate: Rate of anomalies to inject
            noise_level: Level of noise to add
        """
        self.base_data = base_data
        self.anomaly_rate = anomaly_rate
        self.noise_level = noise_level
        self._record_counter = 0
    
    async def generate_stream(
        self,
        count: Optional[int] = None,
        delay: float = 0.1
    ) -> AsyncIterator[StreamRecord]:
        """Generate a stream of synthetic records.
        
        Args:
            count: Number of records to generate (None for infinite)
            delay: Delay between records in seconds
            
        Yields:
            StreamRecord instances
        """
        generated = 0
        
        while count is None or generated < count:
            record = self._generate_record()
            yield record
            
            generated += 1
            if delay > 0:
                await asyncio.sleep(delay)
    
    def _generate_record(self) -> StreamRecord:
        """Generate a single synthetic record."""
        self._record_counter += 1
        
        # Generate base data
        if self.base_data is not None and len(self.base_data) > 0:
            # Sample from base data with variations
            sample_idx = np.random.randint(0, len(self.base_data))
            base_row = self.base_data.iloc[sample_idx].to_dict()
            
            # Add noise
            data = {}
            for key, value in base_row.items():
                if isinstance(value, (int, float)):
                    noise = np.random.normal(0, abs(value) * self.noise_level)
                    data[key] = value + noise
                else:
                    data[key] = value
        else:
            # Generate random data
            data = {
                "feature_1": np.random.normal(0, 1),
                "feature_2": np.random.normal(0, 1),
                "feature_3": np.random.exponential(1),
                "feature_4": np.random.uniform(-1, 1)
            }
        
        # Inject anomalies
        if np.random.random() < self.anomaly_rate:
            self._inject_anomaly(data)
        
        return StreamRecord(
            id=f"record_{self._record_counter}",
            timestamp=datetime.now(),
            data=data,
            metadata={"generated": True, "counter": self._record_counter}
        )
    
    def _inject_anomaly(self, data: Dict[str, Any]) -> None:
        """Inject anomaly into record data."""
        # Simple anomaly injection: multiply random feature by large factor
        numeric_keys = [k for k, v in data.items() if isinstance(v, (int, float))]
        if numeric_keys:
            anomaly_key = np.random.choice(numeric_keys)
            data[anomaly_key] *= np.random.uniform(5, 10)  # Make it anomalous