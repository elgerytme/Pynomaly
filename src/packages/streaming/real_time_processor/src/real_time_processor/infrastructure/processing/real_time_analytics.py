#!/usr/bin/env python3
"""
Real-Time Analytics Engine
Advanced real-time analytics with windowing, aggregations, and pattern detection.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple, AsyncGenerator
from dataclasses import dataclass, asdict, field
from enum import Enum
from collections import defaultdict, deque
import uuid
import statistics
import numpy as np
from concurrent.futures import ThreadPoolExecutor

import redis.asyncio as redis
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge
from kafka import KafkaConsumer, KafkaProducer


# Metrics
ANALYTICS_PROCESSED = Counter('analytics_events_processed_total', 'Total processed analytics events', ['window_type', 'status'])
ANALYTICS_TIME = Histogram('analytics_processing_seconds', 'Time spent on analytics processing', ['operation'])
ACTIVE_WINDOWS = Gauge('analytics_active_windows', 'Number of active analytics windows', ['window_type'])
PATTERN_DETECTIONS = Counter('analytics_patterns_detected_total', 'Total patterns detected', ['pattern_type'])
ANOMALY_DETECTIONS = Counter('analytics_anomalies_detected_total', 'Total anomalies detected', ['anomaly_type'])


class WindowType(Enum):
    """Window types for analytics."""
    TUMBLING = "tumbling"
    SLIDING = "sliding"
    SESSION = "session"
    COUNT = "count"


class AggregationType(Enum):
    """Aggregation types."""
    SUM = "sum"
    COUNT = "count"
    AVERAGE = "average"
    MIN = "min"
    MAX = "max"
    PERCENTILE = "percentile"
    DISTINCT_COUNT = "distinct_count"
    FIRST = "first"
    LAST = "last"


class PatternType(Enum):
    """Pattern detection types."""
    THRESHOLD_BREACH = "threshold_breach"
    TREND_DETECTION = "trend_detection"
    CORRELATION = "correlation"
    SEQUENCE = "sequence"
    FREQUENCY_ANALYSIS = "frequency_analysis"


@dataclass
class AnalyticsEvent:
    """Analytics event structure."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    event_type: str = ""
    source: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    dimensions: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WindowConfig:
    """Window configuration."""
    name: str
    window_type: WindowType
    size_seconds: Optional[int] = None
    size_count: Optional[int] = None
    slide_seconds: Optional[int] = None
    session_timeout_seconds: Optional[int] = None
    key_field: Optional[str] = None  # Field to group by
    enable_late_data: bool = True
    late_data_threshold_seconds: int = 300


@dataclass
class AggregationConfig:
    """Aggregation configuration."""
    name: str
    aggregation_type: AggregationType
    field: str
    percentile: Optional[float] = None  # For percentile aggregation


@dataclass
class PatternConfig:
    """Pattern detection configuration."""
    name: str
    pattern_type: PatternType
    parameters: Dict[str, Any] = field(default_factory=dict)
    threshold: Optional[float] = None
    enabled: bool = True


class Window:
    """Analytics window for time-based aggregations."""
    
    def __init__(self, config: WindowConfig, start_time: datetime, end_time: datetime):
        self.config = config
        self.start_time = start_time
        self.end_time = end_time
        self.events: List[AnalyticsEvent] = []
        self.aggregations: Dict[str, Any] = {}
        self.is_closed = False
        self.last_update = datetime.utcnow()
    
    def add_event(self, event: AnalyticsEvent) -> bool:
        """Add event to window if it belongs."""
        if self.is_closed:
            return False
        
        if self.config.window_type == WindowType.SESSION:
            # Session window - extend end time if within timeout
            time_since_last = (event.timestamp - self.last_update).total_seconds()
            if time_since_last <= self.config.session_timeout_seconds:
                self.end_time = event.timestamp + timedelta(seconds=self.config.session_timeout_seconds)
                self.events.append(event)
                self.last_update = event.timestamp
                return True
            return False
        
        # Time-based windows
        if self.start_time <= event.timestamp < self.end_time:
            self.events.append(event)
            self.last_update = event.timestamp
            return True
        
        return False
    
    def should_close(self, current_time: datetime) -> bool:
        """Check if window should be closed."""
        if self.is_closed:
            return False
        
        if self.config.window_type == WindowType.SESSION:
            return current_time >= self.end_time
        
        return current_time >= self.end_time
    
    def close(self) -> None:
        """Close the window."""
        self.is_closed = True
    
    def get_event_count(self) -> int:
        """Get number of events in window."""
        return len(self.events)


class RealTimeAnalyticsEngine:
    """Advanced real-time analytics engine."""
    
    def __init__(self, kafka_config: Dict[str, Any], redis_url: str = "redis://localhost:6379/2"):
        self.kafka_config = kafka_config
        self.redis_url = redis_url
        
        # Initialize components
        self.redis_client = None
        self.kafka_consumer = None
        self.kafka_producer = None
        
        # Analytics configuration
        self.window_configs: Dict[str, WindowConfig] = {}
        self.aggregation_configs: Dict[str, List[AggregationConfig]] = {}
        self.pattern_configs: Dict[str, PatternConfig] = {}
        
        # Runtime state
        self.active_windows: Dict[str, List[Window]] = defaultdict(list)
        self.pattern_detectors: Dict[str, Callable] = {}
        self.event_buffer: deque = deque(maxlen=10000)
        self.is_running = False
        self.worker_tasks: List[asyncio.Task] = []
        
        # Monitoring and logging
        self.logger = logging.getLogger("analytics_engine")
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Analytics state
        self.metrics_cache: Dict[str, Any] = {}
        self.pattern_state: Dict[str, Any] = {}
        self.anomaly_baselines: Dict[str, Dict[str, float]] = {}
    
    async def initialize(self) -> None:
        """Initialize the analytics engine."""
        try:
            self.logger.info("Initializing real-time analytics engine...")
            
            # Initialize Redis
            self.redis_client = redis.Redis.from_url(self.redis_url)
            await self.redis_client.ping()
            
            # Initialize Kafka consumer
            self.kafka_consumer = KafkaConsumer(
                **self.kafka_config,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='latest',
                enable_auto_commit=True,
                group_id='analytics_engine'
            )
            
            # Initialize Kafka producer for results
            self.kafka_producer = KafkaProducer(
                **self.kafka_config,
                value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None
            )
            
            # Initialize pattern detectors
            self._initialize_pattern_detectors()
            
            self.logger.info("Real-time analytics engine initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize analytics engine: {e}")
            raise
    
    def register_window(self, config: WindowConfig) -> None:
        """Register a window configuration."""
        self.window_configs[config.name] = config
        self.active_windows[config.name] = []
        self.logger.info(f"Registered window: {config.name} ({config.window_type.value})")
    
    def register_aggregation(self, window_name: str, config: AggregationConfig) -> None:
        """Register an aggregation for a window."""
        if window_name not in self.aggregation_configs:
            self.aggregation_configs[window_name] = []
        self.aggregation_configs[window_name].append(config)
        self.logger.info(f"Registered aggregation: {config.name} for window {window_name}")
    
    def register_pattern(self, config: PatternConfig) -> None:
        """Register a pattern detection configuration."""
        self.pattern_configs[config.name] = config
        self.logger.info(f"Registered pattern detector: {config.name} ({config.pattern_type.value})")
    
    def _initialize_pattern_detectors(self) -> None:
        """Initialize pattern detection functions."""
        self.pattern_detectors = {
            PatternType.THRESHOLD_BREACH: self._detect_threshold_breach,
            PatternType.TREND_DETECTION: self._detect_trend,
            PatternType.CORRELATION: self._detect_correlation,
            PatternType.SEQUENCE: self._detect_sequence,
            PatternType.FREQUENCY_ANALYSIS: self._analyze_frequency
        }
    
    async def start(self) -> None:
        """Start the analytics engine."""
        if self.is_running:
            self.logger.warning("Analytics engine is already running")
            return
        
        self.is_running = True
        self.logger.info("Starting real-time analytics engine...")
        
        # Start worker tasks
        self.worker_tasks = [
            asyncio.create_task(self._event_processor()),
            asyncio.create_task(self._window_manager()),
            asyncio.create_task(self._pattern_detector()),
            asyncio.create_task(self._anomaly_detector()),
            asyncio.create_task(self._metrics_publisher()),
            asyncio.create_task(self._cleanup_worker())
        ]
        
        self.logger.info(f"Started {len(self.worker_tasks)} analytics worker tasks")
    
    async def _event_processor(self) -> None:
        """Process incoming events."""
        self.logger.info("Started event processor")
        
        while self.is_running:
            try:
                # Poll for messages
                message_batch = self.kafka_consumer.poll(timeout_ms=1000)
                
                for topic_partition, messages in message_batch.items():
                    for message in messages:
                        event = self._parse_event(message)
                        if event:
                            await self._process_event(event)
                
            except Exception as e:
                self.logger.error(f"Error in event processor: {e}")
                await asyncio.sleep(1)
    
    def _parse_event(self, message) -> Optional[AnalyticsEvent]:
        """Parse Kafka message into analytics event."""
        try:
            data = message.value
            
            event = AnalyticsEvent(
                timestamp=datetime.fromisoformat(data.get('timestamp', datetime.utcnow().isoformat())),
                event_type=data.get('event_type', 'unknown'),
                source=data.get('source', 'unknown'),
                data=data.get('data', {}),
                dimensions=data.get('dimensions', {}),
                metrics=data.get('metrics', {}),
                metadata=data.get('metadata', {})
            )
            
            return event
            
        except Exception as e:
            self.logger.error(f"Failed to parse event: {e}")
            return None
    
    async def _process_event(self, event: AnalyticsEvent) -> None:
        """Process a single analytics event."""
        try:
            # Add to buffer for pattern detection
            self.event_buffer.append(event)
            
            # Process through all configured windows
            for window_name, window_config in self.window_configs.items():
                await self._add_event_to_windows(window_name, event)
            
            ANALYTICS_PROCESSED.labels(window_type="all", status="processed").inc()
            
        except Exception as e:
            self.logger.error(f"Error processing event: {e}")
            ANALYTICS_PROCESSED.labels(window_type="all", status="error").inc()
    
    async def _add_event_to_windows(self, window_name: str, event: AnalyticsEvent) -> None:
        """Add event to appropriate windows."""
        window_config = self.window_configs[window_name]
        windows = self.active_windows[window_name]
        
        # Filter by key field if configured
        if window_config.key_field:
            if window_config.key_field not in event.dimensions:
                return
        
        event_added = False
        
        # Try to add to existing windows
        for window in windows:
            if window.add_event(event):
                event_added = True
                break
        
        # Create new window if needed
        if not event_added:
            new_window = await self._create_window(window_config, event.timestamp)
            if new_window and new_window.add_event(event):
                windows.append(new_window)
                ACTIVE_WINDOWS.labels(window_type=window_config.window_type.value).inc()
        
        # Update metrics
        ACTIVE_WINDOWS.labels(window_type=window_config.window_type.value).set(len(windows))
    
    async def _create_window(self, config: WindowConfig, event_time: datetime) -> Optional[Window]:
        """Create a new window."""
        try:
            if config.window_type == WindowType.TUMBLING:
                # Align to window boundaries
                window_start = self._align_to_window_boundary(event_time, config.size_seconds)
                window_end = window_start + timedelta(seconds=config.size_seconds)
            
            elif config.window_type == WindowType.SLIDING:
                window_start = event_time
                window_end = event_time + timedelta(seconds=config.size_seconds)
            
            elif config.window_type == WindowType.SESSION:
                window_start = event_time
                window_end = event_time + timedelta(seconds=config.session_timeout_seconds)
            
            elif config.window_type == WindowType.COUNT:
                # Count-based window
                window_start = event_time
                window_end = event_time + timedelta(days=1)  # Long end time for count windows
            
            else:
                return None
            
            return Window(config, window_start, window_end)
            
        except Exception as e:
            self.logger.error(f"Failed to create window: {e}")
            return None
    
    def _align_to_window_boundary(self, timestamp: datetime, window_size_seconds: int) -> datetime:
        """Align timestamp to window boundary."""
        epoch = datetime(1970, 1, 1)
        seconds_since_epoch = (timestamp - epoch).total_seconds()
        aligned_seconds = (seconds_since_epoch // window_size_seconds) * window_size_seconds
        return epoch + timedelta(seconds=aligned_seconds)
    
    async def _window_manager(self) -> None:
        """Manage window lifecycle."""
        self.logger.info("Started window manager")
        
        while self.is_running:
            try:
                current_time = datetime.utcnow()
                
                for window_name, windows in self.active_windows.items():
                    windows_to_remove = []
                    
                    for window in windows:
                        if window.should_close(current_time):
                            await self._close_window(window_name, window)
                            windows_to_remove.append(window)
                    
                    # Remove closed windows
                    for window in windows_to_remove:
                        windows.remove(window)
                        ACTIVE_WINDOWS.labels(window_type=window.config.window_type.value).dec()
                
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                self.logger.error(f"Error in window manager: {e}")
                await asyncio.sleep(5)
    
    async def _close_window(self, window_name: str, window: Window) -> None:
        """Close a window and compute aggregations."""
        try:
            start_time = time.time()
            
            window.close()
            
            # Compute aggregations
            aggregations = await self._compute_aggregations(window_name, window)
            
            # Store results
            await self._store_window_results(window_name, window, aggregations)
            
            # Publish results
            await self._publish_window_results(window_name, window, aggregations)
            
            processing_time = time.time() - start_time
            ANALYTICS_TIME.labels(operation="window_close").observe(processing_time)
            
            self.logger.debug(f"Closed window {window_name} with {window.get_event_count()} events")
            
        except Exception as e:
            self.logger.error(f"Error closing window {window_name}: {e}")
    
    async def _compute_aggregations(self, window_name: str, window: Window) -> Dict[str, Any]:
        """Compute aggregations for a window."""
        aggregations = {}
        
        if window_name not in self.aggregation_configs:
            return aggregations
        
        for agg_config in self.aggregation_configs[window_name]:
            try:
                result = await self._compute_single_aggregation(window, agg_config)
                aggregations[agg_config.name] = result
            except Exception as e:
                self.logger.error(f"Error computing aggregation {agg_config.name}: {e}")
        
        return aggregations
    
    async def _compute_single_aggregation(self, window: Window, config: AggregationConfig) -> Any:
        """Compute a single aggregation."""
        events = window.events
        
        if not events:
            return None
        
        # Extract values
        values = []
        for event in events:
            if config.field in event.metrics:
                values.append(event.metrics[config.field])
            elif config.field in event.data:
                try:
                    values.append(float(event.data[config.field]))
                except (ValueError, TypeError):
                    continue
        
        if not values:
            return None
        
        # Compute aggregation
        if config.aggregation_type == AggregationType.SUM:
            return sum(values)
        elif config.aggregation_type == AggregationType.COUNT:
            return len(values)
        elif config.aggregation_type == AggregationType.AVERAGE:
            return statistics.mean(values)
        elif config.aggregation_type == AggregationType.MIN:
            return min(values)
        elif config.aggregation_type == AggregationType.MAX:
            return max(values)
        elif config.aggregation_type == AggregationType.PERCENTILE:
            if config.percentile:
                return np.percentile(values, config.percentile)
        elif config.aggregation_type == AggregationType.DISTINCT_COUNT:
            return len(set(values))
        elif config.aggregation_type == AggregationType.FIRST:
            return values[0] if values else None
        elif config.aggregation_type == AggregationType.LAST:
            return values[-1] if values else None
        
        return None
    
    async def _store_window_results(self, window_name: str, window: Window, aggregations: Dict[str, Any]) -> None:
        """Store window results in Redis."""
        try:
            result_data = {
                "window_name": window_name,
                "window_type": window.config.window_type.value,
                "start_time": window.start_time.isoformat(),
                "end_time": window.end_time.isoformat(),
                "event_count": window.get_event_count(),
                "aggregations": aggregations,
                "computed_at": datetime.utcnow().isoformat()
            }
            
            # Store with TTL of 24 hours
            key = f"analytics_results:{window_name}:{window.start_time.isoformat()}"
            await self.redis_client.setex(key, 86400, json.dumps(result_data, default=str))
            
        except Exception as e:
            self.logger.error(f"Failed to store window results: {e}")
    
    async def _publish_window_results(self, window_name: str, window: Window, aggregations: Dict[str, Any]) -> None:
        """Publish window results to Kafka."""
        try:
            result_message = {
                "window_name": window_name,
                "window_type": window.config.window_type.value,
                "start_time": window.start_time.isoformat(),
                "end_time": window.end_time.isoformat(),
                "event_count": window.get_event_count(),
                "aggregations": aggregations,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            self.kafka_producer.send(
                f"analytics_results_{window_name}",
                value=result_message,
                key=window_name
            )
            
        except Exception as e:
            self.logger.error(f"Failed to publish window results: {e}")
    
    async def _pattern_detector(self) -> None:
        """Pattern detection worker."""
        self.logger.info("Started pattern detector")
        
        while self.is_running:
            try:
                for pattern_name, pattern_config in self.pattern_configs.items():
                    if pattern_config.enabled:
                        await self._detect_pattern(pattern_name, pattern_config)
                
                await asyncio.sleep(10)  # Check patterns every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error in pattern detector: {e}")
                await asyncio.sleep(30)
    
    async def _detect_pattern(self, pattern_name: str, config: PatternConfig) -> None:
        """Detect a specific pattern."""
        try:
            detector = self.pattern_detectors.get(config.pattern_type)
            if not detector:
                return
            
            detection_result = await detector(pattern_name, config)
            
            if detection_result:
                await self._handle_pattern_detection(pattern_name, config, detection_result)
                PATTERN_DETECTIONS.labels(pattern_type=config.pattern_type.value).inc()
            
        except Exception as e:
            self.logger.error(f"Error detecting pattern {pattern_name}: {e}")
    
    async def _detect_threshold_breach(self, pattern_name: str, config: PatternConfig) -> Optional[Dict[str, Any]]:
        """Detect threshold breaches."""
        threshold = config.threshold
        field = config.parameters.get('field')
        
        if not threshold or not field:
            return None
        
        # Check recent events
        recent_events = list(self.event_buffer)[-100:]  # Last 100 events
        
        for event in recent_events:
            value = event.metrics.get(field) or event.data.get(field)
            if value and float(value) > threshold:
                return {
                    "pattern_type": "threshold_breach",
                    "field": field,
                    "value": value,
                    "threshold": threshold,
                    "event": asdict(event)
                }
        
        return None
    
    async def _detect_trend(self, pattern_name: str, config: PatternConfig) -> Optional[Dict[str, Any]]:
        """Detect trends in data."""
        field = config.parameters.get('field')
        window_size = config.parameters.get('window_size', 10)
        
        if not field:
            return None
        
        # Get recent values
        recent_events = list(self.event_buffer)[-window_size:]
        values = []
        
        for event in recent_events:
            value = event.metrics.get(field) or event.data.get(field)
            if value:
                values.append(float(value))
        
        if len(values) < 3:
            return None
        
        # Simple trend detection using linear regression slope
        x = list(range(len(values)))
        slope = np.polyfit(x, values, 1)[0]
        
        trend_threshold = config.parameters.get('trend_threshold', 0.1)
        
        if abs(slope) > trend_threshold:
            return {
                "pattern_type": "trend_detection",
                "field": field,
                "slope": slope,
                "direction": "increasing" if slope > 0 else "decreasing",
                "values": values
            }
        
        return None
    
    async def _detect_correlation(self, pattern_name: str, config: PatternConfig) -> Optional[Dict[str, Any]]:
        """Detect correlations between fields."""
        field1 = config.parameters.get('field1')
        field2 = config.parameters.get('field2')
        window_size = config.parameters.get('window_size', 20)
        
        if not field1 or not field2:
            return None
        
        # Get recent values
        recent_events = list(self.event_buffer)[-window_size:]
        values1, values2 = [], []
        
        for event in recent_events:
            val1 = event.metrics.get(field1) or event.data.get(field1)
            val2 = event.metrics.get(field2) or event.data.get(field2)
            
            if val1 and val2:
                values1.append(float(val1))
                values2.append(float(val2))
        
        if len(values1) < 5:
            return None
        
        # Calculate correlation
        correlation = np.corrcoef(values1, values2)[0, 1]
        correlation_threshold = config.parameters.get('correlation_threshold', 0.8)
        
        if abs(correlation) > correlation_threshold:
            return {
                "pattern_type": "correlation",
                "field1": field1,
                "field2": field2,
                "correlation": correlation,
                "strength": "strong" if abs(correlation) > 0.8 else "moderate"
            }
        
        return None
    
    async def _detect_sequence(self, pattern_name: str, config: PatternConfig) -> Optional[Dict[str, Any]]:
        """Detect event sequences."""
        sequence = config.parameters.get('sequence', [])
        window_size = config.parameters.get('window_size', 50)
        
        if not sequence:
            return None
        
        # Check for sequence in recent events
        recent_events = list(self.event_buffer)[-window_size:]
        event_types = [event.event_type for event in recent_events]
        
        # Simple sequence matching
        for i in range(len(event_types) - len(sequence) + 1):
            if event_types[i:i+len(sequence)] == sequence:
                return {
                    "pattern_type": "sequence",
                    "sequence": sequence,
                    "found_at_index": i,
                    "matching_events": recent_events[i:i+len(sequence)]
                }
        
        return None
    
    async def _analyze_frequency(self, pattern_name: str, config: PatternConfig) -> Optional[Dict[str, Any]]:
        """Analyze event frequency patterns."""
        field = config.parameters.get('field')
        window_seconds = config.parameters.get('window_seconds', 300)
        expected_frequency = config.parameters.get('expected_frequency')
        
        if not field or not expected_frequency:
            return None
        
        # Count events in time window
        cutoff_time = datetime.utcnow() - timedelta(seconds=window_seconds)
        recent_events = [e for e in self.event_buffer if e.timestamp >= cutoff_time]
        
        # Count by field value
        frequency_counts = defaultdict(int)
        for event in recent_events:
            value = event.data.get(field) or event.dimensions.get(field)
            if value:
                frequency_counts[value] += 1
        
        # Check for frequency anomalies
        for value, count in frequency_counts.items():
            frequency = count / (window_seconds / 60)  # per minute
            
            if abs(frequency - expected_frequency) > expected_frequency * 0.5:  # 50% deviation
                return {
                    "pattern_type": "frequency_analysis",
                    "field": field,
                    "value": value,
                    "actual_frequency": frequency,
                    "expected_frequency": expected_frequency,
                    "deviation": abs(frequency - expected_frequency) / expected_frequency
                }
        
        return None
    
    async def _handle_pattern_detection(self, pattern_name: str, config: PatternConfig, result: Dict[str, Any]) -> None:
        """Handle pattern detection result."""
        try:
            detection_message = {
                "pattern_name": pattern_name,
                "pattern_type": config.pattern_type.value,
                "detection_result": result,
                "timestamp": datetime.utcnow().isoformat(),
                "confidence": 1.0  # Could be calculated based on pattern strength
            }
            
            # Store in Redis
            key = f"pattern_detection:{pattern_name}:{datetime.utcnow().isoformat()}"
            await self.redis_client.setex(key, 3600, json.dumps(detection_message, default=str))
            
            # Publish to Kafka
            self.kafka_producer.send(
                "pattern_detections",
                value=detection_message,
                key=pattern_name
            )
            
            self.logger.info(f"Pattern detected: {pattern_name} - {config.pattern_type.value}")
            
        except Exception as e:
            self.logger.error(f"Failed to handle pattern detection: {e}")
    
    async def _anomaly_detector(self) -> None:
        """Anomaly detection worker."""
        self.logger.info("Started anomaly detector")
        
        while self.is_running:
            try:
                await self._detect_anomalies()
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in anomaly detector: {e}")
                await asyncio.sleep(60)
    
    async def _detect_anomalies(self) -> None:
        """Detect anomalies in the data."""
        try:
            # Statistical anomaly detection on recent events
            window_size = 100
            recent_events = list(self.event_buffer)[-window_size:]
            
            if len(recent_events) < 10:
                return
            
            # Group by metric fields
            metric_values = defaultdict(list)
            
            for event in recent_events:
                for metric_name, value in event.metrics.items():
                    if isinstance(value, (int, float)):
                        metric_values[metric_name].append(value)
            
            # Detect anomalies using z-score
            for metric_name, values in metric_values.items():
                if len(values) < 10:
                    continue
                
                mean_val = statistics.mean(values)
                std_val = statistics.stdev(values) if len(values) > 1 else 0
                
                if std_val == 0:
                    continue
                
                # Check last few values for anomalies
                for i, value in enumerate(values[-5:]):
                    z_score = abs(value - mean_val) / std_val
                    
                    if z_score > 3:  # 3-sigma rule
                        await self._handle_anomaly_detection(
                            metric_name, value, mean_val, z_score, recent_events[-(5-i)]
                        )
            
        except Exception as e:
            self.logger.error(f"Error detecting anomalies: {e}")
    
    async def _handle_anomaly_detection(self, metric_name: str, value: float, baseline: float, 
                                      z_score: float, event: AnalyticsEvent) -> None:
        """Handle anomaly detection."""
        try:
            anomaly_message = {
                "anomaly_type": "statistical",
                "metric_name": metric_name,
                "anomalous_value": value,
                "baseline_value": baseline,
                "z_score": z_score,
                "event": asdict(event),
                "timestamp": datetime.utcnow().isoformat(),
                "severity": "high" if z_score > 4 else "medium"
            }
            
            # Store in Redis
            key = f"anomaly_detection:{metric_name}:{datetime.utcnow().isoformat()}"
            await self.redis_client.setex(key, 3600, json.dumps(anomaly_message, default=str))
            
            # Publish to Kafka
            self.kafka_producer.send(
                "anomaly_detections",
                value=anomaly_message,
                key=metric_name
            )
            
            ANOMALY_DETECTIONS.labels(anomaly_type="statistical").inc()
            
            self.logger.warning(f"Anomaly detected in {metric_name}: {value} (z-score: {z_score:.2f})")
            
        except Exception as e:
            self.logger.error(f"Failed to handle anomaly detection: {e}")
    
    async def _metrics_publisher(self) -> None:
        """Publish analytics metrics."""
        while self.is_running:
            try:
                # Update cache with current metrics
                self.metrics_cache.update({
                    "timestamp": datetime.utcnow().isoformat(),
                    "active_windows": sum(len(windows) for windows in self.active_windows.values()),
                    "event_buffer_size": len(self.event_buffer),
                    "registered_patterns": len(self.pattern_configs),
                    "is_running": self.is_running
                })
                
                # Store in Redis
                await self.redis_client.setex(
                    "analytics_metrics",
                    300,  # 5 minutes TTL
                    json.dumps(self.metrics_cache)
                )
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                self.logger.error(f"Error publishing metrics: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_worker(self) -> None:
        """Clean up old data and optimize performance."""
        while self.is_running:
            try:
                # Clean up pattern state
                current_time = datetime.utcnow()
                cutoff_time = current_time - timedelta(hours=1)
                
                # Clean pattern state older than 1 hour
                keys_to_remove = []
                for key, data in self.pattern_state.items():
                    if isinstance(data, dict) and 'timestamp' in data:
                        timestamp = datetime.fromisoformat(data['timestamp'])
                        if timestamp < cutoff_time:
                            keys_to_remove.append(key)
                
                for key in keys_to_remove:
                    del self.pattern_state[key]
                
                await asyncio.sleep(3600)  # Clean up every hour
                
            except Exception as e:
                self.logger.error(f"Error in cleanup worker: {e}")
                await asyncio.sleep(3600)
    
    async def stop(self) -> None:
        """Stop the analytics engine."""
        self.logger.info("Stopping real-time analytics engine...")
        self.is_running = False
        
        # Cancel all worker tasks
        for task in self.worker_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        
        # Close connections
        if self.kafka_consumer:
            self.kafka_consumer.close()
        
        if self.kafka_producer:
            self.kafka_producer.close()
        
        if self.redis_client:
            await self.redis_client.close()
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        
        self.logger.info("Real-time analytics engine stopped")
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get analytics engine statistics."""
        return {
            "is_running": self.is_running,
            "window_configs": len(self.window_configs),
            "active_windows": {name: len(windows) for name, windows in self.active_windows.items()},
            "pattern_configs": len(self.pattern_configs),
            "event_buffer_size": len(self.event_buffer),
            "aggregation_configs": {name: len(aggs) for name, aggs in self.aggregation_configs.items()},
            "metrics_cache": self.metrics_cache,
            "worker_tasks": len(self.worker_tasks)
        }


# Example usage
async def create_analytics_engine():
    """Create and configure analytics engine."""
    kafka_config = {
        "bootstrap_servers": ["localhost:9092"],
        "security_protocol": "PLAINTEXT"
    }
    
    engine = RealTimeAnalyticsEngine(kafka_config)
    await engine.initialize()
    
    # Configure windows
    tumbling_window = WindowConfig(
        name="5min_tumbling",
        window_type=WindowType.TUMBLING,
        size_seconds=300
    )
    engine.register_window(tumbling_window)
    
    # Configure aggregations
    engine.register_aggregation(
        "5min_tumbling",
        AggregationConfig("avg_response_time", AggregationType.AVERAGE, "response_time")
    )
    
    # Configure patterns
    engine.register_pattern(PatternConfig(
        name="high_error_rate",
        pattern_type=PatternType.THRESHOLD_BREACH,
        threshold=0.05,
        parameters={"field": "error_rate"}
    ))
    
    return engine