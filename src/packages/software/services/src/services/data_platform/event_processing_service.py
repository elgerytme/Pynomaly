"""Event processing service for anomaly events."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

from monorepo.domain.entities.anomaly_event import (
    AnomalyEvent,
    EventAggregation,
    EventFilter,
    EventPattern,
    EventSeverity,
    EventStatus,
    EventSummary,
    EventType,
)

logger = logging.getLogger(__name__)


class EventProcessingService:
    """Service for processing and managing anomaly events."""

    def __init__(
        self,
        event_repository: Any,  # EventRepositoryProtocol when implemented
        notification_service: Any,  # NotificationService when implemented
        pattern_detector: Any | None = None,  # PatternDetector when implemented
    ):
        """Initialize the event processing service.

        Args:
            event_repository: Event repository
            notification_service: Notification service
            pattern_detector: Pattern processing service
        """
        self.event_repository = event_repository
        self.notification_service = notification_service
        self.pattern_detector = pattern_detector
        self._event_handlers: dict[EventType, list[Callable]] = {}
        self._processing_queue: asyncio.Queue = asyncio.Queue()
        self._worker_tasks: list[asyncio.Task] = []
        self._running = False

    async def start_processing(self, num_workers: int = 4) -> None:
        """Start event processing workers.

        Args:
            num_workers: Number of worker tasks to start
        """
        if self._running:
            return

        self._running = True

        # Start worker tasks
        for i in range(num_workers):
            task = asyncio.create_task(self._event_worker(f"worker-{i}"))
            self._worker_tasks.append(task)

        logger.info(f"Started {num_workers} event processing workers")

    async def stop_processing(self) -> None:
        """Stop event processing workers."""
        if not self._running:
            return

        self._running = False

        # Cancel worker tasks
        for task in self._worker_tasks:
            task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(*self._worker_tasks, return_exceptions=True)

        self._worker_tasks.clear()
        logger.info("Stopped event processing workers")

    async def create_event(self, event: AnomalyEvent) -> AnomalyEvent:
        """Create and queue an event for processing.

        Args:
            event: Event to create

        Returns:
            Created event
        """
        # Store event
        stored_event = await self.event_repository.create_event(event)

        # Queue for processing
        await self._processing_queue.put(stored_event)

        return stored_event

    async def process_event(self, event: AnomalyEvent) -> AnomalyEvent:
        """Process a single event.

        Args:
            event: Event to process

        Returns:
            Processed event
        """
        try:
            # Mark as processing
            event.mark_processing()
            await self.event_repository.update_event(event)

            # Run event handlers
            await self._run_event_handlers(event)

            # Check for patterns
            if self.pattern_detector:
                await self._check_event_patterns(event)

            # Send notifications if needed
            await self._send_event_notifications(event)

            # Mark as processed
            event.mark_processed()
            await self.event_repository.update_event(event)

            return event

        except Exception as e:
            logger.error(f"Error processing event {event.id}: {e}")

            # Mark as failed and schedule retry
            retry_after = datetime.utcnow() + timedelta(minutes=5)
            event.mark_failed(str(e), retry_after)
            await self.event_repository.update_event(event)

            return event

    async def acknowledge_event(
        self, event_id: UUID, user: str, notes: str | None = None
    ) -> AnomalyEvent:
        """Acknowledge an event.

        Args:
            event_id: Event identifier
            user: User acknowledging the event
            notes: Optional acknowledgment notes

        Returns:
            Updated event
        """
        event = await self.event_repository.get_event(event_id)
        if not event:
            raise ValueError(f"Event {event_id} not found")

        event.acknowledge(user, notes)
        updated_event = await self.event_repository.update_event(event)

        return updated_event

    async def resolve_event(
        self, event_id: UUID, user: str, notes: str | None = None
    ) -> AnomalyEvent:
        """Resolve an event.

        Args:
            event_id: Event identifier
            user: User resolving the event
            notes: Optional resolution notes

        Returns:
            Updated event
        """
        event = await self.event_repository.get_event(event_id)
        if not event:
            raise ValueError(f"Event {event_id} not found")

        event.resolve(user, notes)
        updated_event = await self.event_repository.update_event(event)

        return updated_event

    async def ignore_event(
        self, event_id: UUID, user: str, reason: str | None = None
    ) -> AnomalyEvent:
        """Ignore an event.

        Args:
            event_id: Event identifier
            user: User ignoring the event
            reason: Optional ignore reason

        Returns:
            Updated event
        """
        event = await self.event_repository.get_event(event_id)
        if not event:
            raise ValueError(f"Event {event_id} not found")

        event.ignore(user, reason)
        updated_event = await self.event_repository.update_event(event)

        return updated_event

    async def query_events(self, filter_criteria: EventFilter) -> list[AnomalyEvent]:
        """Query events with filters.

        Args:
            filter_criteria: Filter criteria

        Returns:
            Matching events
        """
        return await self.event_repository.query_events(filter_criteria)

    async def get_event_summary(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        detector_ids: list[UUID] | None = None,
    ) -> EventSummary:
        """Get event summary statistics.

        Args:
            start_time: Start time for summary
            end_time: End time for summary
            detector_ids: Filter by detector IDs

        Returns:
            Event summary
        """
        if not start_time:
            start_time = datetime.utcnow() - timedelta(hours=24)
        if not end_time:
            end_time = datetime.utcnow()

        # Get events in time range
        filter_criteria = EventFilter(
            event_time_start=start_time,
            event_time_end=end_time,
            detector_ids=detector_ids,
            limit=10000,  # Large limit to get all events
        )

        events = await self.event_repository.query_events(filter_criteria)

        # Calculate statistics
        total_events = len(events)

        # Events by type
        events_by_type = {}
        for event_type in EventType:
            events_by_type[event_type.value] = len(
                [e for e in events if e.event_type == event_type]
            )

        # Events by severity
        events_by_severity = {}
        for severity in EventSeverity:
            events_by_severity[severity.value] = len(
                [e for e in events if e.severity == severity]
            )

        # Events by status
        events_by_status = {}
        for status in EventStatus:
            events_by_status[status.value] = len(
                [e for e in events if e.status == status]
            )

        # Anomaly statistics
        anomaly_events = [
            e for e in events if e.event_type == EventType.ANOMALY_DETECTED
        ]
        anomaly_rate = len(anomaly_events) / total_events if total_events > 0 else 0.0

        avg_anomaly_score = None
        if anomaly_events:
            scores = [
                e.anomaly_data.anomaly_score for e in anomaly_events if e.anomaly_data
            ]
            avg_anomaly_score = sum(scores) / len(scores) if scores else None

        # Resolution statistics
        resolved_events = [e for e in events if e.status == EventStatus.RESOLVED]
        resolution_rate = (
            len(resolved_events) / total_events if total_events > 0 else 0.0
        )

        avg_resolution_time = None
        if resolved_events:
            resolution_times = []
            for event in resolved_events:
                if event.resolved_at and event.event_time:
                    resolution_time = (
                        event.resolved_at - event.event_time
                    ).total_seconds()
                    resolution_times.append(resolution_time)
            avg_resolution_time = (
                sum(resolution_times) / len(resolution_times)
                if resolution_times
                else None
            )

        # Top detectors
        detector_counts = {}
        for event in events:
            if event.detector_id:
                detector_counts[str(event.detector_id)] = (
                    detector_counts.get(str(event.detector_id), 0) + 1
                )

        top_detectors = [
            {"detector_id": detector_id, "event_count": count}
            for detector_id, count in sorted(
                detector_counts.items(), key=lambda x: x[1], reverse=True
            )[:10]
        ]

        # Top data sources
        source_counts = {}
        for event in events:
            if event.data_source:
                source_counts[event.data_source] = (
                    source_counts.get(event.data_source, 0) + 1
                )

        top_data_sources = [
            {"data_source": source, "event_count": count}
            for source, count in sorted(
                source_counts.items(), key=lambda x: x[1], reverse=True
            )[:10]
        ]

        return EventSummary(
            total_events=total_events,
            events_by_type=events_by_type,
            events_by_severity=events_by_severity,
            events_by_status=events_by_status,
            anomaly_rate=anomaly_rate,
            avg_anomaly_score=avg_anomaly_score,
            resolution_rate=resolution_rate,
            avg_resolution_time=avg_resolution_time,
            top_detectors=top_detectors,
            top_data_sources=top_data_sources,
            time_range={"start": start_time, "end": end_time},
        )

    async def aggregate_events(
        self,
        group_by: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        filter_criteria: EventFilter | None = None,
    ) -> list[EventAggregation]:
        """Aggregate events by specified criteria.

        Args:
            group_by: Field to group by (e.g., 'detector_id', 'data_source', 'severity')
            start_time: Start time for aggregation
            end_time: End time for aggregation
            filter_criteria: Additional filter criteria

        Returns:
            List of event aggregations
        """
        if not start_time:
            start_time = datetime.utcnow() - timedelta(hours=24)
        if not end_time:
            end_time = datetime.utcnow()

        # Create filter if not provided
        if not filter_criteria:
            filter_criteria = EventFilter()

        filter_criteria.event_time_start = start_time
        filter_criteria.event_time_end = end_time
        filter_criteria.limit = 10000  # Large limit

        # Get events
        events = await self.event_repository.query_events(filter_criteria)

        # Group events
        groups = {}
        for event in events:
            # Get group key
            if group_by == "detector_id":
                key = str(event.detector_id) if event.detector_id else "unknown"
            elif group_by == "data_source":
                key = event.data_source or "unknown"
            elif group_by == "severity":
                key = event.severity.value
            elif group_by == "event_type":
                key = event.event_type.value
            else:
                key = "all"

            if key not in groups:
                groups[key] = []
            groups[key].append(event)

        # Create aggregations
        aggregations = []
        for group_key, group_events in groups.items():
            # Calculate statistics
            severities = [e.severity for e in group_events]
            min_severity = min(severities)
            max_severity = max(severities)

            event_times = [e.event_time for e in group_events]
            first_event_time = min(event_times)
            last_event_time = max(event_times)

            # Anomaly score average
            anomaly_events = [e for e in group_events if e.anomaly_data]
            avg_anomaly_score = None
            if anomaly_events:
                scores = [e.anomaly_data.anomaly_score for e in anomaly_events]
                avg_anomaly_score = sum(scores) / len(scores)

            # Unique counts
            unique_detectors = len(
                set(str(e.detector_id) for e in group_events if e.detector_id)
            )
            unique_sessions = len(
                set(
                    str(e.source_session_id)
                    for e in group_events
                    if e.source_session_id
                )
            )

            # Status counts
            resolved_count = len(
                [e for e in group_events if e.status == EventStatus.RESOLVED]
            )
            acknowledged_count = len(
                [e for e in group_events if e.status == EventStatus.ACKNOWLEDGED]
            )

            aggregation = EventAggregation(
                group_key=group_key,
                count=len(group_events),
                min_severity=min_severity,
                max_severity=max_severity,
                first_event_time=first_event_time,
                last_event_time=last_event_time,
                avg_anomaly_score=avg_anomaly_score,
                unique_detectors=unique_detectors,
                unique_sessions=unique_sessions,
                resolved_count=resolved_count,
                acknowledged_count=acknowledged_count,
            )

            aggregations.append(aggregation)

        # Sort by count descending
        aggregations.sort(key=lambda x: x.count, reverse=True)

        return aggregations

    async def create_event_pattern(
        self,
        name: str,
        pattern_type: str,
        conditions: dict[str, Any],
        time_window: int,
        created_by: str,
        description: str | None = None,
        confidence: float = 0.8,
        alert_threshold: int = 1,
    ) -> EventPattern:
        """Create an event pattern for processing.

        Args:
            name: Pattern name
            pattern_type: Type of pattern
            conditions: Pattern conditions
            time_window: Time window in seconds
            created_by: Pattern creator
            description: Pattern description
            confidence: Pattern confidence
            alert_threshold: Alert threshold

        Returns:
            Created event pattern
        """
        pattern = EventPattern(
            name=name,
            description=description,
            pattern_type=pattern_type,
            conditions=conditions,
            time_window=time_window,
            confidence=confidence,
            alert_threshold=alert_threshold,
            created_by=created_by,
        )

        stored_pattern = await self.event_repository.create_pattern(pattern)

        return stored_pattern

    async def detect_event_patterns(
        self, events: list[AnomalyEvent]
    ) -> list[tuple[EventPattern, list[AnomalyEvent]]]:
        """Detect patterns in events.

        Args:
            events: Events to analyze

        Returns:
            List of (pattern, matching_events) tuples
        """
        if not self.pattern_detector:
            return []

        # Get active patterns
        patterns = await self.event_repository.get_active_patterns()

        detected_patterns = []
        for pattern in patterns:
            matching_events = await self._match_pattern(pattern, events)
            if len(matching_events) >= pattern.alert_threshold:
                detected_patterns.append((pattern, matching_events))

                # Update pattern statistics
                pattern.match_count += 1
                pattern.last_matched = datetime.utcnow()
                await self.event_repository.update_pattern(pattern)

        return detected_patterns

    def register_event_handler(
        self, event_type: EventType, handler: Callable[[AnomalyEvent], None]
    ) -> None:
        """Register an event handler for a specific event type.

        Args:
            event_type: Event type to handle
            handler: Handler function
        """
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []

        self._event_handlers[event_type].append(handler)

    def unregister_event_handler(
        self, event_type: EventType, handler: Callable[[AnomalyEvent], None]
    ) -> None:
        """Unregister an event handler.

        Args:
            event_type: Event type
            handler: Handler function to remove
        """
        if event_type in self._event_handlers:
            self._event_handlers[event_type] = [
                h for h in self._event_handlers[event_type] if h != handler
            ]

    async def _event_worker(self, worker_name: str) -> None:
        """Event processing worker."""
        logger.info(f"Started event worker: {worker_name}")

        while self._running:
            try:
                # Get event from queue with timeout
                event = await asyncio.wait_for(
                    self._processing_queue.get(), timeout=1.0
                )

                # Process the event
                await self.process_event(event)

                # Mark task as done
                self._processing_queue.task_done()

            except TimeoutError:
                # No event available, continue
                continue
            except Exception as e:
                logger.error(f"Error in event worker {worker_name}: {e}")
                # Continue processing other events

        logger.info(f"Stopped event worker: {worker_name}")

    async def _run_event_handlers(self, event: AnomalyEvent) -> None:
        """Run registered event handlers."""
        handlers = self._event_handlers.get(event.event_type, [])

        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                logger.error(f"Error in event handler: {e}")

    async def _check_event_patterns(self, event: AnomalyEvent) -> None:
        """Check if event matches any patterns."""
        # Get recent events in pattern time window
        window_start = event.event_time - timedelta(seconds=3600)  # 1 hour window

        filter_criteria = EventFilter(
            event_time_start=window_start,
            event_time_end=event.event_time,
            limit=1000,
        )

        recent_events = await self.event_repository.query_events(filter_criteria)
        recent_events.append(event)  # Include current event

        # Detect patterns
        detected_patterns = await self.detect_event_patterns(recent_events)

        # Process detected patterns
        for pattern, matching_events in detected_patterns:
            if pattern.alert_enabled:
                await self._send_pattern_alert(pattern, matching_events)

    async def _match_pattern(
        self, pattern: EventPattern, events: list[AnomalyEvent]
    ) -> list[AnomalyEvent]:
        """Match events against a pattern."""
        # Simple pattern matching implementation
        # In practice, this would be more sophisticated

        matching_events = []

        if pattern.pattern_type == "frequency":
            # Frequency pattern: multiple events of same type in time window
            event_type = pattern.conditions.get("event_type")
            min_count = pattern.conditions.get("min_count", 1)

            type_events = [e for e in events if e.event_type.value == event_type]
            if len(type_events) >= min_count:
                matching_events = type_events

        elif pattern.pattern_type == "sequence":
            # Sequence pattern: events in specific order
            # This would implement sequence matching logic
            pass

        elif pattern.pattern_type == "correlation":
            # Correlation pattern: related events from different sources
            # This would implement correlation logic
            pass

        return matching_events

    async def _send_event_notifications(self, event: AnomalyEvent) -> None:
        """Send notifications for event if needed."""
        # Determine if notification should be sent
        should_notify = event.is_actionable() or event.severity in [
            EventSeverity.HIGH,
            EventSeverity.CRITICAL,
        ]

        if should_notify and not event.notification_sent:
            # Send notification
            await self.notification_service.send_event_notification(event)

            # Mark as sent
            event.notification_sent = True
            event.notification_channels = ["email", "slack"]  # Example channels
            await self.event_repository.update_event(event)

    async def _send_pattern_alert(
        self, pattern: EventPattern, matching_events: list[AnomalyEvent]
    ) -> None:
        """Send alert for detected pattern."""
        logger.warning(
            f"Pattern detected: {pattern.name} - {len(matching_events)} matching events"
        )

        # This would send actual pattern alerts
        await self.notification_service.send_pattern_alert(pattern, matching_events)
