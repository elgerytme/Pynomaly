"""
Real-Time Analytics Service

Provides real-time processing and analytics capabilities for monitoring data,
enabling instant insights, anomaly detection, and proactive alerting.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
import math
import statistics

import numpy as np
import pandas as pd
from collections import deque, defaultdict
import structlog

from mlops.infrastructure.monitoring.pipeline_monitor import AlertSeverity


class StreamingWindowType(Enum):
    """Types of streaming windows for analytics."""
    TUMBLING = "tumbling"  # Non-overlapping fixed-size windows
    SLIDING = "sliding"    # Overlapping windows
    SESSION = "session"    # Variable-size based on activity


class AggregationType(Enum):
    """Types of aggregations for streaming data."""
    COUNT = "count"
    SUM = "sum"
    AVERAGE = "average"
    MIN = "min"
    MAX = "max"
    MEDIAN = "median"
    PERCENTILE = "percentile"
    STANDARD_DEVIATION = "std"
    VARIANCE = "variance"


@dataclass
class StreamingWindow:
    """Configuration for streaming window."""
    window_type: StreamingWindowType
    window_size_seconds: int
    slide_interval_seconds: Optional[int] = None  # For sliding windows
    session_timeout_seconds: Optional[int] = None  # For session windows
    max_events: Optional[int] = None  # Maximum events in window


@dataclass
class StreamingQuery:
    """Configuration for streaming analytics query."""
    query_id: str
    name: str
    description: str
    source_streams: List[str]
    filters: Dict[str, Any] = field(default_factory=dict)
    aggregations: List[Dict[str, Any]] = field(default_factory=list)
    window: Optional[StreamingWindow] = None
    output_stream: Optional[str] = None
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class StreamingEvent:
    """A single streaming event."""
    stream_id: str
    event_id: str
    timestamp: datetime
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalyticsResult:
    """Result from streaming analytics."""
    query_id: str
    window_start: datetime
    window_end: datetime
    result_data: Dict[str, Any]
    event_count: int
    processing_time_ms: float
    generated_at: datetime = field(default_factory=datetime.utcnow)


class StreamProcessor:
    """Processes streaming data for a specific stream."""
    
    def __init__(self, stream_id: str, buffer_size: int = 1000):
        self.stream_id = stream_id
        self.buffer = deque(maxlen=buffer_size)
        self.event_count = 0
        self.last_event_time: Optional[datetime] = None
        
        # Statistics
        self.events_per_second = 0.0
        self.avg_processing_time = 0.0
        self.error_count = 0
        
        # Window management
        self.windows: Dict[str, List[StreamingEvent]] = defaultdict(list)
        
    def add_event(self, event: StreamingEvent) -> None:
        """Add an event to the stream."""
        self.buffer.append(event)
        self.event_count += 1
        self.last_event_time = event.timestamp
        
        # Update events per second (simple sliding average)
        if len(self.buffer) > 1:
            time_diff = (self.buffer[-1].timestamp - self.buffer[-2].timestamp).total_seconds()
            if time_diff > 0:
                self.events_per_second = 0.9 * self.events_per_second + 0.1 * (1.0 / time_diff)
    
    def get_events_in_window(self, window_start: datetime, window_end: datetime) -> List[StreamingEvent]:
        """Get events within a time window."""
        return [
            event for event in self.buffer
            if window_start <= event.timestamp < window_end
        ]
    
    def get_recent_events(self, seconds: int) -> List[StreamingEvent]:
        """Get events from the last N seconds."""
        cutoff_time = datetime.utcnow() - timedelta(seconds=seconds)
        return [event for event in self.buffer if event.timestamp >= cutoff_time]


class RealTimeAnalyticsService:
    """
    Real-time analytics service for processing streaming monitoring data
    and generating instant insights.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = structlog.get_logger(__name__)
        
        # Stream management
        self.streams: Dict[str, StreamProcessor] = {}
        self.queries: Dict[str, StreamingQuery] = {}
        self.result_handlers: Dict[str, Callable] = {}
        
        # Processing configuration
        self.processing_interval = self.config.get("processing_interval_seconds", 30)
        self.buffer_size = self.config.get("buffer_size", 1000)
        self.enable_streaming = self.config.get("enable_streaming", True)
        
        # Background tasks
        self.processing_tasks: Dict[str, asyncio.Task] = {}
        self.is_running = False
        
        # Analytics state
        self.processing_stats = {
            "total_events_processed": 0,
            "total_queries_executed": 0,
            "avg_processing_latency_ms": 0.0,
            "error_count": 0,
            "last_processing_time": None
        }
        
        # Real-time alerting
        self.alert_handlers: List[Callable] = []
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
    
    async def start(self) -> None:
        """Start the real-time analytics service."""
        if self.is_running:
            self.logger.warning("Real-time analytics service is already running")
            return
        
        try:
            self.is_running = True
            
            # Start query processing loops
            await self._start_query_processors()
            
            # Start monitoring and health checks
            await self._start_monitoring_tasks()
            
            self.logger.info(
                "Real-time analytics service started",
                processing_interval=self.processing_interval,
                buffer_size=self.buffer_size,
                streaming_enabled=self.enable_streaming
            )
            
        except Exception as e:
            self.logger.error(f"Failed to start real-time analytics service: {e}")
            await self.stop()
            raise
    
    async def stop(self) -> None:
        """Stop the real-time analytics service."""
        if not self.is_running:
            return
        
        try:
            self.is_running = False
            
            # Stop all processing tasks
            for task_name, task in self.processing_tasks.items():
                if not task.done():
                    task.cancel()
            
            await asyncio.gather(*self.processing_tasks.values(), return_exceptions=True)
            self.processing_tasks.clear()
            
            self.logger.info("Real-time analytics service stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping real-time analytics service: {e}")
    
    async def _start_query_processors(self) -> None:
        """Start processing loops for all enabled queries."""
        for query_id, query in self.queries.items():
            if query.enabled:
                task = asyncio.create_task(self._query_processing_loop(query))
                self.processing_tasks[f"query_{query_id}"] = task
    
    async def _start_monitoring_tasks(self) -> None:
        """Start monitoring and maintenance tasks."""
        # Stream statistics update task
        stats_task = asyncio.create_task(self._update_statistics_loop())
        self.processing_tasks["statistics_updater"] = stats_task
        
        # Buffer cleanup task
        cleanup_task = asyncio.create_task(self._buffer_cleanup_loop())
        self.processing_tasks["buffer_cleanup"] = cleanup_task
        
        # Alert processing task
        alert_task = asyncio.create_task(self._alert_processing_loop())
        self.processing_tasks["alert_processor"] = alert_task
    
    async def _query_processing_loop(self, query: StreamingQuery) -> None:
        """Processing loop for a specific streaming query."""
        while self.is_running:
            try:
                start_time = datetime.utcnow()
                
                # Process the query
                result = await self._execute_streaming_query(query)
                
                if result:
                    # Calculate processing time
                    processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                    result.processing_time_ms = processing_time
                    
                    # Handle the result
                    await self._handle_query_result(query, result)
                    
                    # Update statistics
                    self.processing_stats["total_queries_executed"] += 1
                    self.processing_stats["last_processing_time"] = datetime.utcnow()
                
                # Wait before next processing cycle
                await asyncio.sleep(self.processing_interval)
                
            except Exception as e:
                self.logger.error(
                    f"Error processing query {query.query_id}: {e}",
                    query_name=query.name
                )
                self.processing_stats["error_count"] += 1
                await asyncio.sleep(10)  # Wait before retry
    
    async def _execute_streaming_query(self, query: StreamingQuery) -> Optional[AnalyticsResult]:
        """Execute a streaming analytics query."""
        try:
            # Get relevant streams
            relevant_events = []
            
            for stream_id in query.source_streams:
                if stream_id in self.streams:
                    stream_processor = self.streams[stream_id]
                    
                    # Get events based on window configuration
                    if query.window:
                        window_events = self._get_windowed_events(stream_processor, query.window)
                    else:
                        # Default: last processing interval
                        window_events = stream_processor.get_recent_events(self.processing_interval)
                    
                    relevant_events.extend(window_events)
            
            if not relevant_events:
                return None
            
            # Apply filters
            filtered_events = self._apply_filters(relevant_events, query.filters)
            
            if not filtered_events:
                return None
            
            # Calculate window boundaries
            if filtered_events:
                window_start = min(event.timestamp for event in filtered_events)
                window_end = max(event.timestamp for event in filtered_events)
            else:
                now = datetime.utcnow()
                window_start = now - timedelta(seconds=self.processing_interval)
                window_end = now
            
            # Execute aggregations
            aggregation_results = {}
            for aggregation in query.aggregations:
                result = self._execute_aggregation(filtered_events, aggregation)
                aggregation_results[aggregation["name"]] = result
            
            return AnalyticsResult(
                query_id=query.query_id,
                window_start=window_start,
                window_end=window_end,
                result_data=aggregation_results,
                event_count=len(filtered_events),
                processing_time_ms=0.0  # Will be set by caller
            )
            
        except Exception as e:
            self.logger.error(f"Error executing query {query.query_id}: {e}")
            return None
    
    def _get_windowed_events(self, stream_processor: StreamProcessor, window: StreamingWindow) -> List[StreamingEvent]:
        """Get events based on window configuration."""
        now = datetime.utcnow()
        
        if window.window_type == StreamingWindowType.TUMBLING:
            # Non-overlapping windows
            window_start = now - timedelta(seconds=window.window_size_seconds)
            return stream_processor.get_events_in_window(window_start, now)
        
        elif window.window_type == StreamingWindowType.SLIDING:
            # Overlapping windows
            slide_interval = window.slide_interval_seconds or window.window_size_seconds
            window_start = now - timedelta(seconds=window.window_size_seconds)
            return stream_processor.get_events_in_window(window_start, now)
        
        elif window.window_type == StreamingWindowType.SESSION:
            # Session-based windows (simplified implementation)
            session_timeout = window.session_timeout_seconds or 300
            return stream_processor.get_recent_events(session_timeout)
        
        else:
            # Default to tumbling window
            return stream_processor.get_recent_events(window.window_size_seconds)
    
    def _apply_filters(self, events: List[StreamingEvent], filters: Dict[str, Any]) -> List[StreamingEvent]:
        """Apply filters to events."""
        if not filters:
            return events
        
        filtered_events = []
        
        for event in events:
            matches = True
            
            for filter_key, filter_value in filters.items():
                if filter_key not in event.data:
                    matches = False
                    break
                
                event_value = event.data[filter_key]
                
                # Support different filter types
                if isinstance(filter_value, dict):
                    # Range or comparison filters
                    if "$gt" in filter_value and event_value <= filter_value["$gt"]:
                        matches = False
                        break
                    if "$lt" in filter_value and event_value >= filter_value["$lt"]:
                        matches = False
                        break
                    if "$eq" in filter_value and event_value != filter_value["$eq"]:
                        matches = False
                        break
                    if "$in" in filter_value and event_value not in filter_value["$in"]:
                        matches = False
                        break
                else:
                    # Direct value comparison
                    if event_value != filter_value:
                        matches = False
                        break
            
            if matches:
                filtered_events.append(event)
        
        return filtered_events
    
    def _execute_aggregation(self, events: List[StreamingEvent], aggregation: Dict[str, Any]) -> Any:
        """Execute an aggregation on events."""
        agg_type = AggregationType(aggregation["type"])
        field = aggregation.get("field")
        
        if not events:
            return None
        
        # Extract values for the field
        if field:
            values = [event.data.get(field) for event in events if field in event.data]
            # Filter out None values
            values = [v for v in values if v is not None]
        else:
            values = [1] * len(events)  # For count operations
        
        if not values:
            return None
        
        try:
            if agg_type == AggregationType.COUNT:
                return len(values)
            elif agg_type == AggregationType.SUM:
                return sum(values)
            elif agg_type == AggregationType.AVERAGE:
                return statistics.mean(values)
            elif agg_type == AggregationType.MIN:
                return min(values)
            elif agg_type == AggregationType.MAX:
                return max(values)
            elif agg_type == AggregationType.MEDIAN:
                return statistics.median(values)
            elif agg_type == AggregationType.PERCENTILE:
                percentile = aggregation.get("percentile", 95)
                return np.percentile(values, percentile)
            elif agg_type == AggregationType.STANDARD_DEVIATION:
                return statistics.stdev(values) if len(values) > 1 else 0
            elif agg_type == AggregationType.VARIANCE:
                return statistics.variance(values) if len(values) > 1 else 0
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Error executing aggregation {agg_type}: {e}")
            return None
    
    async def _handle_query_result(self, query: StreamingQuery, result: AnalyticsResult) -> None:
        """Handle the result of a streaming query."""
        # Send to output stream if configured
        if query.output_stream:
            await self.publish_event(query.output_stream, {
                "query_id": query.query_id,
                "query_name": query.name,
                "result": result.result_data,
                "window_start": result.window_start.isoformat(),
                "window_end": result.window_end.isoformat(),
                "event_count": result.event_count,
                "processing_time_ms": result.processing_time_ms
            })
        
        # Call result handlers
        if query.query_id in self.result_handlers:
            try:
                await self.result_handlers[query.query_id](result)
            except Exception as e:
                self.logger.error(f"Error calling result handler for query {query.query_id}: {e}")
        
        # Check for alert conditions
        await self._check_alert_conditions(query, result)
    
    async def _check_alert_conditions(self, query: StreamingQuery, result: AnalyticsResult) -> None:
        """Check if query result triggers any alerts."""
        for alert_rule_id, alert_rule in self.alert_rules.items():
            if query.query_id in alert_rule.get("query_ids", []):
                if self._evaluate_alert_condition(result, alert_rule):
                    await self._trigger_alert(alert_rule, result)
    
    def _evaluate_alert_condition(self, result: AnalyticsResult, alert_rule: Dict[str, Any]) -> bool:
        """Evaluate if an alert condition is met."""
        condition = alert_rule.get("condition", {})
        field = condition.get("field")
        operator = condition.get("operator")
        threshold = condition.get("threshold")
        
        if not all([field, operator, threshold]):
            return False
        
        value = result.result_data.get(field)
        if value is None:
            return False
        
        if operator == "gt":
            return value > threshold
        elif operator == "lt":
            return value < threshold
        elif operator == "eq":
            return value == threshold
        elif operator == "gte":
            return value >= threshold
        elif operator == "lte":
            return value <= threshold
        else:
            return False
    
    async def _trigger_alert(self, alert_rule: Dict[str, Any], result: AnalyticsResult) -> None:
        """Trigger an alert based on query result."""
        alert_data = {
            "alert_rule_id": alert_rule["id"],
            "alert_name": alert_rule["name"],
            "severity": alert_rule.get("severity", "medium"),
            "message": alert_rule.get("message", "Alert condition met"),
            "query_id": result.query_id,
            "result_data": result.result_data,
            "triggered_at": datetime.utcnow().isoformat()
        }
        
        # Call alert handlers
        for handler in self.alert_handlers:
            try:
                await handler(alert_data)
            except Exception as e:
                self.logger.error(f"Error calling alert handler: {e}")
        
        self.logger.warning(
            "Real-time alert triggered",
            alert_rule_id=alert_rule["id"],
            alert_name=alert_rule["name"],
            severity=alert_rule.get("severity")
        )
    
    async def _update_statistics_loop(self) -> None:
        """Update service statistics periodically."""
        while self.is_running:
            try:
                # Update processing statistics
                total_events = sum(stream.event_count for stream in self.streams.values())
                self.processing_stats["total_events_processed"] = total_events
                
                # Update average events per second across all streams
                if self.streams:
                    avg_eps = statistics.mean([stream.events_per_second for stream in self.streams.values()])
                    self.processing_stats["avg_events_per_second"] = avg_eps
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error updating statistics: {e}")
                await asyncio.sleep(60)
    
    async def _buffer_cleanup_loop(self) -> None:
        """Clean up old data from stream buffers."""
        while self.is_running:
            try:
                # Clean up old events (keep last hour by default)
                retention_seconds = self.config.get("event_retention_seconds", 3600)
                cutoff_time = datetime.utcnow() - timedelta(seconds=retention_seconds)
                
                for stream in self.streams.values():
                    # Remove old events from buffer
                    while stream.buffer and stream.buffer[0].timestamp < cutoff_time:
                        stream.buffer.popleft()
                
                await asyncio.sleep(300)  # Cleanup every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error during buffer cleanup: {e}")
                await asyncio.sleep(300)
    
    async def _alert_processing_loop(self) -> None:
        """Process real-time alerts and notifications."""
        while self.is_running:
            try:
                # Process any pending alerts or notifications
                # This could integrate with external alerting systems
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error processing alerts: {e}")
                await asyncio.sleep(30)
    
    async def publish_event(self, stream_id: str, event_data: Dict[str, Any]) -> None:
        """Publish an event to a stream."""
        if not self.enable_streaming:
            return
        
        # Create or get stream processor
        if stream_id not in self.streams:
            self.streams[stream_id] = StreamProcessor(stream_id, self.buffer_size)
        
        # Create event
        event = StreamingEvent(
            stream_id=stream_id,
            event_id=f"{stream_id}_{datetime.utcnow().timestamp()}",
            timestamp=datetime.utcnow(),
            data=event_data
        )
        
        # Add to stream
        self.streams[stream_id].add_event(event)
        
        self.logger.debug(
            "Event published to stream",
            stream_id=stream_id,
            event_id=event.event_id
        )
    
    async def register_query(self, query_config: Dict[str, Any]) -> str:
        """Register a new streaming query."""
        query = StreamingQuery(
            query_id=query_config["query_id"],
            name=query_config["name"],
            description=query_config.get("description", ""),
            source_streams=query_config["source_streams"],
            filters=query_config.get("filters", {}),
            aggregations=query_config.get("aggregations", []),
            window=StreamingWindow(**query_config["window"]) if "window" in query_config else None,
            output_stream=query_config.get("output_stream"),
            enabled=query_config.get("enabled", True)
        )
        
        self.queries[query.query_id] = query
        
        # Start processing loop for the query if service is running
        if self.is_running and query.enabled:
            task = asyncio.create_task(self._query_processing_loop(query))
            self.processing_tasks[f"query_{query.query_id}"] = task
        
        self.logger.info(
            "Streaming query registered",
            query_id=query.query_id,
            query_name=query.name
        )
        
        return query.query_id
    
    def register_result_handler(self, query_id: str, handler: Callable) -> None:
        """Register a result handler for a query."""
        self.result_handlers[query_id] = handler
    
    def add_alert_handler(self, handler: Callable) -> None:
        """Add an alert handler."""
        self.alert_handlers.append(handler)
    
    async def register_alert_rule(self, alert_rule: Dict[str, Any]) -> str:
        """Register a real-time alert rule."""
        rule_id = alert_rule["id"]
        self.alert_rules[rule_id] = alert_rule
        
        self.logger.info(
            "Alert rule registered",
            rule_id=rule_id,
            rule_name=alert_rule.get("name")
        )
        
        return rule_id
    
    async def get_stream_status(self, stream_id: str = None) -> Dict[str, Any]:
        """Get status of streams."""
        if stream_id:
            if stream_id in self.streams:
                stream = self.streams[stream_id]
                return {
                    "stream_id": stream_id,
                    "event_count": stream.event_count,
                    "events_per_second": stream.events_per_second,
                    "buffer_size": len(stream.buffer),
                    "last_event_time": stream.last_event_time.isoformat() if stream.last_event_time else None,
                    "error_count": stream.error_count
                }
            else:
                return {"error": "Stream not found"}
        else:
            return {
                "total_streams": len(self.streams),
                "total_events": sum(stream.event_count for stream in self.streams.values()),
                "streams": [
                    {
                        "stream_id": sid,
                        "event_count": stream.event_count,
                        "events_per_second": stream.events_per_second,
                        "buffer_size": len(stream.buffer)
                    }
                    for sid, stream in self.streams.items()
                ]
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        return {
            "status": "healthy" if self.is_running else "stopped",
            "service_uptime": "running" if self.is_running else "stopped",
            "active_streams": len(self.streams),
            "active_queries": len([q for q in self.queries.values() if q.enabled]),
            "processing_stats": self.processing_stats,
            "last_updated": datetime.utcnow().isoformat()
        }