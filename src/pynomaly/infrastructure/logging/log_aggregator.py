"""Log aggregation and streaming system."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, AsyncGenerator, Callable
from uuid import uuid4

from .structured_logger import LogLevel, StructuredLogger


class LogStreamType(Enum):
    """Types of log streams."""
    REALTIME = "realtime"
    BATCH = "batch"
    FILTERED = "filtered"
    AGGREGATED = "aggregated"


@dataclass
class LogEntry:
    """Individual log entry."""
    id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    level: str = "INFO"
    logger_name: str = ""
    message: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    source: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "level": self.level,
            "logger_name": self.logger_name,
            "message": self.message,
            "context": self.context,
            "tags": self.tags,
            "source": self.source
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LogEntry:
        """Create from dictionary."""
        return cls(
            id=data.get("id", str(uuid4())),
            timestamp=datetime.fromisoformat(data["timestamp"]) if isinstance(data.get("timestamp"), str) else data.get("timestamp", datetime.utcnow()),
            level=data.get("level", "INFO"),
            logger_name=data.get("logger_name", ""),
            message=data.get("message", ""),
            context=data.get("context", {}),
            tags=data.get("tags", []),
            source=data.get("source", "")
        )


@dataclass
class LogFilter:
    """Filter configuration for log streams."""
    levels: list[str] = field(default_factory=list)
    loggers: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    message_patterns: list[str] = field(default_factory=list)
    time_range: tuple[datetime, datetime] | None = None
    max_entries: int | None = None
    
    def matches(self, entry: LogEntry) -> bool:
        """Check if log entry matches filter."""
        # Level filter
        if self.levels and entry.level not in self.levels:
            return False
        
        # Logger filter
        if self.loggers and not any(logger in entry.logger_name for logger in self.loggers):
            return False
        
        # Tags filter
        if self.tags and not any(tag in entry.tags for tag in self.tags):
            return False
        
        # Message pattern filter
        if self.message_patterns:
            import re
            if not any(re.search(pattern, entry.message, re.IGNORECASE) for pattern in self.message_patterns):
                return False
        
        # Time range filter
        if self.time_range:
            start_time, end_time = self.time_range
            if not (start_time <= entry.timestamp <= end_time):
                return False
        
        return True


@dataclass
class LogAggregation:
    """Aggregated log statistics."""
    time_window: timedelta
    start_time: datetime
    end_time: datetime
    total_entries: int = 0
    level_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    logger_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    tag_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    error_patterns: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "time_window_seconds": self.time_window.total_seconds(),
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "total_entries": self.total_entries,
            "level_counts": dict(self.level_counts),
            "logger_counts": dict(self.logger_counts),
            "tag_counts": dict(self.tag_counts),
            "error_patterns": dict(self.error_patterns)
        }


class LogStream:
    """Represents a stream of log entries."""
    
    def __init__(
        self,
        name: str,
        stream_type: LogStreamType = LogStreamType.REALTIME,
        filter_config: LogFilter | None = None,
        buffer_size: int = 1000,
        batch_size: int = 100,
        batch_timeout: float = 5.0
    ):
        """Initialize log stream.
        
        Args:
            name: Stream name
            stream_type: Type of stream
            filter_config: Filter configuration
            buffer_size: Maximum entries in buffer
            batch_size: Batch size for batch streams
            batch_timeout: Timeout for batch collection
        """
        self.name = name
        self.stream_type = stream_type
        self.filter_config = filter_config or LogFilter()
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        
        # Stream state
        self._buffer: deque[LogEntry] = deque(maxlen=buffer_size)
        self._subscribers: list[Callable[[LogEntry], None]] = []
        self._batch_subscribers: list[Callable[[list[LogEntry]], None]] = []
        self._lock = threading.RLock()
        
        # Batch processing
        self._batch_buffer: list[LogEntry] = []
        self._last_batch_time = time.time()
        self._batch_timer: threading.Timer | None = None
        
        # Statistics
        self.stats = {
            "entries_processed": 0,
            "entries_filtered": 0,
            "subscribers_count": 0,
            "batch_subscribers_count": 0,
            "last_entry_time": None
        }
    
    def add_entry(self, entry: LogEntry):
        """Add log entry to stream."""
        with self._lock:
            # Apply filter
            if not self.filter_config.matches(entry):
                self.stats["entries_filtered"] += 1
                return
            
            # Add to buffer
            self._buffer.append(entry)
            self.stats["entries_processed"] += 1
            self.stats["last_entry_time"] = entry.timestamp
            
            # Notify subscribers based on stream type
            if self.stream_type == LogStreamType.REALTIME:
                self._notify_subscribers(entry)
            elif self.stream_type == LogStreamType.BATCH:
                self._add_to_batch(entry)
    
    def _notify_subscribers(self, entry: LogEntry):
        """Notify real-time subscribers."""
        for subscriber in self._subscribers:
            try:
                subscriber(entry)
            except Exception as e:
                print(f"Error notifying subscriber: {e}")
    
    def _add_to_batch(self, entry: LogEntry):
        """Add entry to batch."""
        self._batch_buffer.append(entry)
        
        # Check if batch is ready
        if len(self._batch_buffer) >= self.batch_size:
            self._flush_batch()
        elif not self._batch_timer:
            # Start batch timer
            self._batch_timer = threading.Timer(self.batch_timeout, self._flush_batch)
            self._batch_timer.start()
    
    def _flush_batch(self):
        """Flush current batch to subscribers."""
        if not self._batch_buffer:
            return
        
        batch = self._batch_buffer.copy()
        self._batch_buffer.clear()
        
        if self._batch_timer:
            self._batch_timer.cancel()
            self._batch_timer = None
        
        # Notify batch subscribers
        for subscriber in self._batch_subscribers:
            try:
                subscriber(batch)
            except Exception as e:
                print(f"Error notifying batch subscriber: {e}")
    
    def subscribe(self, callback: Callable[[LogEntry], None]):
        """Subscribe to real-time log entries."""
        with self._lock:
            self._subscribers.append(callback)
            self.stats["subscribers_count"] = len(self._subscribers)
    
    def subscribe_batch(self, callback: Callable[[list[LogEntry]], None]):
        """Subscribe to batch log entries."""
        with self._lock:
            self._batch_subscribers.append(callback)
            self.stats["batch_subscribers_count"] = len(self._batch_subscribers)
    
    def unsubscribe(self, callback: Callable[[LogEntry], None]):
        """Unsubscribe from real-time log entries."""
        with self._lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)
                self.stats["subscribers_count"] = len(self._subscribers)
    
    def unsubscribe_batch(self, callback: Callable[[list[LogEntry]], None]):
        """Unsubscribe from batch log entries."""
        with self._lock:
            if callback in self._batch_subscribers:
                self._batch_subscribers.remove(callback)
                self.stats["batch_subscribers_count"] = len(self._batch_subscribers)
    
    def get_recent_entries(self, count: int = 100) -> list[LogEntry]:
        """Get recent log entries."""
        with self._lock:
            return list(self._buffer)[-count:]
    
    def get_entries_since(self, since: datetime) -> list[LogEntry]:
        """Get entries since a specific time."""
        with self._lock:
            return [entry for entry in self._buffer if entry.timestamp >= since]
    
    def clear(self):
        """Clear stream buffer."""
        with self._lock:
            self._buffer.clear()
            self._batch_buffer.clear()
            if self._batch_timer:
                self._batch_timer.cancel()
                self._batch_timer = None
    
    def get_stats(self) -> dict[str, Any]:
        """Get stream statistics."""
        with self._lock:
            return {
                "name": self.name,
                "type": self.stream_type.value,
                "buffer_size": len(self._buffer),
                "batch_buffer_size": len(self._batch_buffer),
                **self.stats
            }


class LogAggregator:
    """Aggregates and manages log streams."""
    
    def __init__(
        self,
        storage_path: Path | None = None,
        max_streams: int = 100,
        default_buffer_size: int = 1000,
        aggregation_interval: int = 300,  # 5 minutes
        enable_persistence: bool = True
    ):
        """Initialize log aggregator.
        
        Args:
            storage_path: Path for log storage
            max_streams: Maximum number of streams
            default_buffer_size: Default buffer size for streams
            aggregation_interval: Interval for log aggregation (seconds)
            enable_persistence: Whether to enable log persistence
        """
        self.storage_path = storage_path
        self.max_streams = max_streams
        self.default_buffer_size = default_buffer_size
        self.aggregation_interval = aggregation_interval
        self.enable_persistence = enable_persistence
        
        # Streams management
        self._streams: dict[str, LogStream] = {}
        self._lock = threading.RLock()
        
        # Aggregation
        self._aggregations: dict[str, LogAggregation] = {}
        self._aggregation_thread: threading.Thread | None = None
        self._shutdown_event = threading.Event()
        
        # Persistence
        self._persistence_thread: threading.Thread | None = None
        self._persistence_queue: deque[LogEntry] = deque()
        
        # Statistics
        self.stats = {
            "total_entries": 0,
            "streams_count": 0,
            "aggregations_count": 0,
            "persistence_errors": 0,
            "last_aggregation_time": None
        }
        
        # Start background tasks
        if self.aggregation_interval > 0:
            self._start_aggregation_thread()
        
        if self.enable_persistence and self.storage_path:
            self._start_persistence_thread()
    
    def create_stream(
        self,
        name: str,
        stream_type: LogStreamType = LogStreamType.REALTIME,
        filter_config: LogFilter | None = None,
        **kwargs
    ) -> LogStream:
        """Create a new log stream."""
        with self._lock:
            if len(self._streams) >= self.max_streams:
                raise ValueError(f"Maximum number of streams ({self.max_streams}) reached")
            
            if name in self._streams:
                raise ValueError(f"Stream '{name}' already exists")
            
            stream = LogStream(
                name=name,
                stream_type=stream_type,
                filter_config=filter_config,
                buffer_size=kwargs.get("buffer_size", self.default_buffer_size),
                **kwargs
            )
            
            self._streams[name] = stream
            self.stats["streams_count"] = len(self._streams)
            
            return stream
    
    def get_stream(self, name: str) -> LogStream | None:
        """Get stream by name."""
        with self._lock:
            return self._streams.get(name)
    
    def remove_stream(self, name: str) -> bool:
        """Remove stream by name."""
        with self._lock:
            if name in self._streams:
                stream = self._streams.pop(name)
                stream.clear()
                self.stats["streams_count"] = len(self._streams)
                return True
            return False
    
    def add_log_entry(self, entry: LogEntry):
        """Add log entry to all streams."""
        with self._lock:
            self.stats["total_entries"] += 1
            
            # Add to persistence queue
            if self.enable_persistence:
                self._persistence_queue.append(entry)
            
            # Distribute to streams
            for stream in self._streams.values():
                stream.add_entry(entry)
    
    def add_log_from_record(self, record: dict[str, Any]):
        """Add log entry from log record."""
        entry = LogEntry(
            timestamp=datetime.fromisoformat(record.get("timestamp", datetime.utcnow().isoformat())),
            level=record.get("level", "INFO"),
            logger_name=record.get("logger_name", ""),
            message=record.get("message", ""),
            context=record.get("context", {}),
            tags=record.get("tags", []),
            source=record.get("source", "")
        )
        self.add_log_entry(entry)
    
    def _start_aggregation_thread(self):
        """Start background aggregation thread."""
        def aggregation_worker():
            while not self._shutdown_event.wait(self.aggregation_interval):
                try:
                    self._perform_aggregation()
                except Exception as e:
                    print(f"Error in aggregation: {e}")
        
        self._aggregation_thread = threading.Thread(target=aggregation_worker, daemon=True)
        self._aggregation_thread.start()
    
    def _perform_aggregation(self):
        """Perform log aggregation."""
        now = datetime.utcnow()
        window_start = now - timedelta(seconds=self.aggregation_interval)
        
        aggregation = LogAggregation(
            time_window=timedelta(seconds=self.aggregation_interval),
            start_time=window_start,
            end_time=now
        )
        
        # Collect statistics from all streams
        with self._lock:
            for stream in self._streams.values():
                entries = stream.get_entries_since(window_start)
                
                for entry in entries:
                    aggregation.total_entries += 1
                    aggregation.level_counts[entry.level] += 1
                    aggregation.logger_counts[entry.logger_name] += 1
                    
                    for tag in entry.tags:
                        aggregation.tag_counts[tag] += 1
                    
                    # Track error patterns
                    if entry.level in ["ERROR", "CRITICAL"]:
                        # Simple error pattern extraction
                        error_key = f"{entry.logger_name}:{entry.level}"
                        aggregation.error_patterns[error_key] += 1
        
        # Store aggregation
        aggregation_key = f"{window_start.isoformat()}_{now.isoformat()}"
        self._aggregations[aggregation_key] = aggregation
        
        # Cleanup old aggregations (keep last 100)
        if len(self._aggregations) > 100:
            oldest_keys = sorted(self._aggregations.keys())[:len(self._aggregations) - 100]
            for key in oldest_keys:
                del self._aggregations[key]
        
        self.stats["aggregations_count"] = len(self._aggregations)
        self.stats["last_aggregation_time"] = now
    
    def _start_persistence_thread(self):
        """Start background persistence thread."""
        def persistence_worker():
            while not self._shutdown_event.is_set():
                try:
                    self._persist_logs()
                    time.sleep(1)  # Check every second
                except Exception as e:
                    self.stats["persistence_errors"] += 1
                    print(f"Error in log persistence: {e}")
        
        self._persistence_thread = threading.Thread(target=persistence_worker, daemon=True)
        self._persistence_thread.start()
    
    def _persist_logs(self):
        """Persist logs to storage."""
        if not self.storage_path or not self._persistence_queue:
            return
        
        # Collect batch of entries
        entries_to_persist = []
        while self._persistence_queue and len(entries_to_persist) < 1000:
            entries_to_persist.append(self._persistence_queue.popleft())
        
        if not entries_to_persist:
            return
        
        # Ensure storage directory exists
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create filename with timestamp
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = self.storage_path.parent / f"logs_{timestamp}_{uuid4().hex[:8]}.jsonl"
        
        # Write entries as JSON lines
        with open(filename, 'w') as f:
            for entry in entries_to_persist:
                f.write(json.dumps(entry.to_dict()) + '\n')
    
    def get_aggregations(self, since: datetime | None = None) -> list[LogAggregation]:
        """Get log aggregations."""
        with self._lock:
            aggregations = list(self._aggregations.values())
            if since:
                aggregations = [agg for agg in aggregations if agg.end_time >= since]
            return sorted(aggregations, key=lambda x: x.start_time)
    
    def get_stats(self) -> dict[str, Any]:
        """Get aggregator statistics."""
        with self._lock:
            return {
                "aggregator_stats": self.stats,
                "streams": {name: stream.get_stats() for name, stream in self._streams.items()},
                "persistence_queue_size": len(self._persistence_queue),
                "aggregations_count": len(self._aggregations)
            }
    
    def shutdown(self):
        """Shutdown aggregator."""
        self._shutdown_event.set()
        
        # Final persistence
        if self.enable_persistence:
            try:
                self._persist_logs()
            except Exception:
                pass
        
        # Wait for threads
        if self._aggregation_thread and self._aggregation_thread.is_alive():
            self._aggregation_thread.join(timeout=5)
        
        if self._persistence_thread and self._persistence_thread.is_alive():
            self._persistence_thread.join(timeout=5)
        
        # Clear streams
        with self._lock:
            for stream in self._streams.values():
                stream.clear()
            self._streams.clear()


# Global aggregator instance
_global_aggregator: LogAggregator | None = None


def get_global_aggregator() -> LogAggregator:
    """Get or create global log aggregator."""
    global _global_aggregator
    if _global_aggregator is None:
        _global_aggregator = LogAggregator()
    return _global_aggregator


def configure_log_aggregation(
    storage_path: Path | None = None,
    max_streams: int = 100,
    aggregation_interval: int = 300,
    enable_persistence: bool = True
) -> LogAggregator:
    """Configure global log aggregation."""
    global _global_aggregator
    _global_aggregator = LogAggregator(
        storage_path=storage_path,
        max_streams=max_streams,
        aggregation_interval=aggregation_interval,
        enable_persistence=enable_persistence
    )
    return _global_aggregator