"""Event processing API endpoints."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from pynomaly.application.services.event_processing_service import (
    EventProcessingService,
)
from pynomaly.domain.entities.anomaly_event import (
    AnomalyEvent,
    EventAggregation,
    EventFilter,
    EventPattern,
    EventSeverity,
    EventStatus,
    EventSummary,
    EventType,
)
from pynomaly.infrastructure.config import Container
from pynomaly.presentation.api.deps import get_container
from pynomaly.presentation.api.docs.response_models import (
    ErrorResponse,
    HTTPResponses,
    SuccessResponse,
)

router = APIRouter(
    prefix="/events",
    tags=["Event Processing"],
    responses={
        401: HTTPResponses.unauthorized_401(),
        403: HTTPResponses.forbidden_403(),
        404: HTTPResponses.not_found_404(),
        500: HTTPResponses.server_error_500(),
    },
)


class AcknowledgeEventRequest(BaseModel):
    """Request for acknowledging an event."""

    notes: str | None = Field(None, description="Acknowledgment notes")


class ResolveEventRequest(BaseModel):
    """Request for resolving an event."""

    notes: str | None = Field(None, description="Resolution notes")


class IgnoreEventRequest(BaseModel):
    """Request for ignoring an event."""

    reason: str | None = Field(None, description="Reason for ignoring")


class CreatePatternRequest(BaseModel):
    """Request for creating event pattern."""

    name: str = Field(..., description="Pattern name")
    pattern_type: str = Field(
        ..., description="Pattern type (frequency, sequence, correlation)"
    )
    conditions: dict[str, Any] = Field(..., description="Pattern matching conditions")
    time_window_seconds: int = Field(
        ..., description="Time window for pattern detection"
    )
    description: str | None = Field(None, description="Pattern description")
    confidence: float = Field(default=0.8, description="Pattern confidence score")
    alert_threshold: int = Field(
        default=1, description="Number of matches before alerting"
    )


class EventQueryRequest(BaseModel):
    """Request for querying events."""

    event_types: list[EventType] | None = Field(
        None, description="Filter by event types"
    )
    severities: list[EventSeverity] | None = Field(
        None, description="Filter by severities"
    )
    statuses: list[EventStatus] | None = Field(None, description="Filter by statuses")
    detector_ids: list[UUID] | None = Field(None, description="Filter by detector IDs")
    session_ids: list[UUID] | None = Field(None, description="Filter by session IDs")
    data_sources: list[str] | None = Field(None, description="Filter by data sources")

    event_time_start: datetime | None = Field(None, description="Event time start")
    event_time_end: datetime | None = Field(None, description="Event time end")

    title_contains: str | None = Field(None, description="Filter by title content")
    description_contains: str | None = Field(
        None, description="Filter by description content"
    )
    tags: list[str] | None = Field(None, description="Filter by tags")
    correlation_id: str | None = Field(None, description="Filter by correlation ID")

    min_anomaly_score: float | None = Field(None, description="Minimum anomaly score")
    max_anomaly_score: float | None = Field(None, description="Maximum anomaly score")
    min_confidence: float | None = Field(None, description="Minimum confidence")

    limit: int = Field(default=100, description="Maximum number of results")
    offset: int = Field(default=0, description="Result offset")
    sort_by: str = Field(default="event_time", description="Sort field")
    sort_order: str = Field(default="desc", description="Sort order (asc, desc)")


async def get_event_service(
    container: Container = Depends(get_container),
) -> EventProcessingService:
    """Get event processing service."""
    # This would be properly injected in a real implementation
    return EventProcessingService(
        event_repository=None,  # Would be injected
        notification_service=None,  # Would be injected
        pattern_detector=None,  # Would be injected
    )


@router.post(
    "/query",
    response_model=SuccessResponse[list[AnomalyEvent]],
    summary="Query Events",
    description="""
    Query anomaly events with comprehensive filtering options.
    
    This endpoint provides powerful querying capabilities to find events based on:
    
    **Event Classification:**
    - **Types**: Filter by event types (anomaly_detected, data_quality_issue, etc.)
    - **Severities**: Filter by severity levels (info, low, medium, high, critical)
    - **Statuses**: Filter by processing status (pending, processed, resolved, etc.)
    
    **Source Filtering:**
    - **Detectors**: Find events from specific anomaly detectors
    - **Sessions**: Find events from specific streaming sessions
    - **Data Sources**: Filter by original data source systems
    
    **Content Filtering:**
    - **Text Search**: Search in event titles and descriptions
    - **Tags**: Filter by event tags
    - **Correlation**: Find related events using correlation IDs
    
    **Anomaly-Specific Filtering:**
    - **Score Range**: Filter by anomaly score thresholds
    - **Confidence**: Filter by detection confidence levels
    
    **Time-Based Filtering:**
    - **Event Time**: When the original event occurred
    - **Ingestion Time**: When the event was received by the system
    
    **Use Cases:**
    - Security incident investigation
    - Performance troubleshooting
    - Quality assurance monitoring
    - Compliance auditing
    - Operational dashboards
    
    **Example Query:**
    ```json
    {
      "event_types": ["anomaly_detected"],
      "severities": ["high", "critical"],
      "event_time_start": "2024-12-25T00:00:00Z",
      "event_time_end": "2024-12-25T23:59:59Z",
      "min_anomaly_score": 0.8,
      "limit": 50
    }
    ```
    """,
)
async def query_events(
    query: EventQueryRequest,
    event_service: EventProcessingService = Depends(get_event_service),
) -> SuccessResponse[list[AnomalyEvent]]:
    """Query events with filters."""
    try:
        filter_criteria = EventFilter(
            event_types=query.event_types,
            severities=query.severities,
            statuses=query.statuses,
            detector_ids=query.detector_ids,
            session_ids=query.session_ids,
            data_sources=query.data_sources,
            event_time_start=query.event_time_start,
            event_time_end=query.event_time_end,
            title_contains=query.title_contains,
            description_contains=query.description_contains,
            tags=query.tags,
            correlation_id=query.correlation_id,
            min_anomaly_score=query.min_anomaly_score,
            max_anomaly_score=query.max_anomaly_score,
            min_confidence=query.min_confidence,
            limit=query.limit,
            offset=query.offset,
            sort_by=query.sort_by,
            sort_order=query.sort_order,
        )

        events = await event_service.query_events(filter_criteria)

        return SuccessResponse(
            data=events, message=f"Found {len(events)} events matching criteria"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to query events: {str(e)}")


@router.get(
    "/{event_id}",
    response_model=SuccessResponse[AnomalyEvent],
    summary="Get Event Details",
    description="""
    Get detailed information about a specific event.
    
    Returns complete event information including:
    - Event classification and metadata
    - Anomaly detection details (if applicable)
    - Source information and context
    - Processing history and status
    - Related events and correlations
    """,
)
async def get_event(
    event_id: UUID, event_service: EventProcessingService = Depends(get_event_service)
) -> SuccessResponse[AnomalyEvent]:
    """Get a specific event by ID."""
    try:
        # This would be implemented in the event service
        # For now, return a mock response
        raise HTTPException(
            status_code=501, detail="Event retrieval not yet implemented"
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get event: {str(e)}")


@router.post(
    "/{event_id}/acknowledge",
    response_model=SuccessResponse[AnomalyEvent],
    summary="Acknowledge Event",
    description="""
    Acknowledge an event to indicate it has been reviewed.
    
    Acknowledging an event:
    - Changes status to `acknowledged`
    - Records who acknowledged it and when
    - Optionally includes acknowledgment notes
    - Stops further escalation alerts
    
    **Use Cases:**
    - Incident response workflows
    - Manual review processes
    - Escalation management
    - Audit trail maintenance
    """,
)
async def acknowledge_event(
    event_id: UUID,
    request: AcknowledgeEventRequest,
    user: str = Query(..., description="User acknowledging the event"),
    event_service: EventProcessingService = Depends(get_event_service),
) -> SuccessResponse[AnomalyEvent]:
    """Acknowledge an event."""
    try:
        event = await event_service.acknowledge_event(event_id, user, request.notes)

        return SuccessResponse(data=event, message="Event acknowledged successfully")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to acknowledge event: {str(e)}"
        )


@router.post(
    "/{event_id}/resolve",
    response_model=SuccessResponse[AnomalyEvent],
    summary="Resolve Event",
    description="""
    Resolve an event to indicate the issue has been addressed.
    
    Resolving an event:
    - Changes status to `resolved`
    - Records who resolved it and when
    - Includes resolution notes explaining the action taken
    - Closes the event in monitoring systems
    
    **Best Practices:**
    - Include detailed resolution notes
    - Document root cause if identified
    - Reference any corrective actions taken
    - Link to related tickets or documentation
    """,
)
async def resolve_event(
    event_id: UUID,
    request: ResolveEventRequest,
    user: str = Query(..., description="User resolving the event"),
    event_service: EventProcessingService = Depends(get_event_service),
) -> SuccessResponse[AnomalyEvent]:
    """Resolve an event."""
    try:
        event = await event_service.resolve_event(event_id, user, request.notes)

        return SuccessResponse(data=event, message="Event resolved successfully")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to resolve event: {str(e)}"
        )


@router.post(
    "/{event_id}/ignore",
    response_model=SuccessResponse[AnomalyEvent],
    summary="Ignore Event",
    description="""
    Ignore an event to indicate it should be disregarded.
    
    Ignoring an event:
    - Changes status to `ignored`
    - Records who ignored it and when
    - Includes reason for ignoring
    - Prevents further processing and alerts
    
    **Common Reasons:**
    - False positive detection
    - Known issue with accepted risk
    - Duplicate or redundant event
    - Test data or maintenance activity
    """,
)
async def ignore_event(
    event_id: UUID,
    request: IgnoreEventRequest,
    user: str = Query(..., description="User ignoring the event"),
    event_service: EventProcessingService = Depends(get_event_service),
) -> SuccessResponse[AnomalyEvent]:
    """Ignore an event."""
    try:
        event = await event_service.ignore_event(event_id, user, request.reason)

        return SuccessResponse(data=event, message="Event ignored successfully")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to ignore event: {str(e)}")


@router.get(
    "/summary",
    response_model=SuccessResponse[EventSummary],
    summary="Get Event Summary",
    description="""
    Get summary statistics for events within a time range.
    
    **Summary Includes:**
    - **Total Counts**: Events by type, severity, and status
    - **Quality Metrics**: Anomaly rate, average scores, resolution rate
    - **Performance**: Average resolution time, processing efficiency
    - **Top Sources**: Most active detectors and data sources
    - **Trends**: Time-based patterns and distributions
    
    **Use Cases:**
    - Executive dashboards and reporting
    - System health monitoring
    - Performance trend analysis
    - Capacity planning
    - Quality assessment
    
    **Time Ranges:**
    - Last 24 hours (default)
    - Custom date ranges
    - Real-time rolling windows
    """,
)
async def get_event_summary(
    start_time: datetime | None = Query(None, description="Summary start time"),
    end_time: datetime | None = Query(None, description="Summary end time"),
    detector_ids: list[UUID] | None = Query(None, description="Filter by detector IDs"),
    event_service: EventProcessingService = Depends(get_event_service),
) -> SuccessResponse[EventSummary]:
    """Get event summary statistics."""
    try:
        summary = await event_service.get_event_summary(
            start_time=start_time,
            end_time=end_time,
            detector_ids=detector_ids,
        )

        return SuccessResponse(
            data=summary, message="Retrieved event summary successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get summary: {str(e)}")


@router.post(
    "/aggregate",
    response_model=SuccessResponse[list[EventAggregation]],
    summary="Aggregate Events",
    description="""
    Aggregate events by specified criteria for analysis.
    
    **Aggregation Options:**
    - **detector_id**: Group by anomaly detector
    - **data_source**: Group by original data source
    - **severity**: Group by event severity level
    - **event_type**: Group by event type
    - **status**: Group by processing status
    
    **Aggregation Statistics:**
    - Event counts and percentages
    - Severity distribution (min/max)
    - Time range (first/last event)
    - Unique source counts
    - Resolution and acknowledgment rates
    - Average anomaly scores
    
    **Use Cases:**
    - Performance analysis by detector
    - Data quality assessment by source
    - Operational metrics by severity
    - Trend analysis over time
    - Resource allocation planning
    
    **Example:**
    Group events by detector to see which detectors are most active:
    ```json
    {
      "group_by": "detector_id",
      "start_time": "2024-12-24T00:00:00Z",
      "end_time": "2024-12-25T00:00:00Z"
    }
    ```
    """,
)
async def aggregate_events(
    group_by: str = Query(..., description="Field to group by"),
    start_time: datetime | None = Query(None, description="Aggregation start time"),
    end_time: datetime | None = Query(None, description="Aggregation end time"),
    event_service: EventProcessingService = Depends(get_event_service),
) -> SuccessResponse[list[EventAggregation]]:
    """Aggregate events by specified criteria."""
    try:
        aggregations = await event_service.aggregate_events(
            group_by=group_by,
            start_time=start_time,
            end_time=end_time,
        )

        return SuccessResponse(
            data=aggregations,
            message=f"Generated {len(aggregations)} aggregations grouped by {group_by}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to aggregate events: {str(e)}"
        )


@router.post(
    "/patterns",
    response_model=SuccessResponse[EventPattern],
    summary="Create Event Pattern",
    description="""
    Create a pattern for automatic event detection and alerting.
    
    Event patterns help identify recurring issues and trends by:
    - **Frequency Patterns**: Multiple events of same type in time window
    - **Sequence Patterns**: Events occurring in specific order
    - **Correlation Patterns**: Related events from different sources
    
    **Pattern Types:**
    
    **Frequency Pattern Example:**
    ```json
    {
      "name": "High Error Rate",
      "pattern_type": "frequency",
      "conditions": {
        "event_type": "anomaly_detected",
        "min_count": 5,
        "severity": "high"
      },
      "time_window_seconds": 300
    }
    ```
    
    **Sequence Pattern Example:**
    ```json
    {
      "name": "System Degradation",
      "pattern_type": "sequence",
      "conditions": {
        "sequence": [
          {"event_type": "performance_degradation"},
          {"event_type": "anomaly_detected"},
          {"event_type": "system_alert"}
        ]
      },
      "time_window_seconds": 600
    }
    ```
    
    **Benefits:**
    - Early warning systems
    - Automated incident detection
    - Trend identification
    - Proactive alerting
    - Root cause analysis
    """,
    responses={
        201: HTTPResponses.created_201("Pattern created successfully"),
        400: HTTPResponses.bad_request_400("Invalid pattern configuration"),
    },
)
async def create_event_pattern(
    request: CreatePatternRequest,
    created_by: str = Query(..., description="User creating the pattern"),
    event_service: EventProcessingService = Depends(get_event_service),
) -> SuccessResponse[EventPattern]:
    """Create an event pattern."""
    try:
        pattern = await event_service.create_event_pattern(
            name=request.name,
            pattern_type=request.pattern_type,
            conditions=request.conditions,
            time_window=request.time_window_seconds,
            created_by=created_by,
            description=request.description,
            confidence=request.confidence,
            alert_threshold=request.alert_threshold,
        )

        return SuccessResponse(
            data=pattern, message="Event pattern created successfully"
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to create pattern: {str(e)}"
        )


@router.get(
    "/patterns/detect",
    response_model=SuccessResponse[list[dict[str, Any]]],
    summary="Detect Event Patterns",
    description="""
    Detect patterns in recent events using configured pattern definitions.
    
    This endpoint analyzes recent events to identify:
    - Matching patterns based on frequency, sequence, or correlation
    - Pattern confidence scores and match counts
    - Affected events and time ranges
    - Recommended actions and alerts
    
    **Detection Process:**
    1. Retrieve recent events within analysis window
    2. Apply all active pattern definitions
    3. Calculate pattern match confidence
    4. Generate alerts for patterns exceeding thresholds
    
    **Response Format:**
    ```json
    [
      {
        "pattern": { "name": "High Error Rate", ... },
        "matching_events": [ ... ],
        "confidence": 0.95,
        "match_count": 7,
        "time_range": { "start": "...", "end": "..." }
      }
    ]
    ```
    
    **Use Cases:**
    - Real-time pattern monitoring
    - Incident investigation
    - Trend analysis
    - Automated alerting
    """,
)
async def detect_event_patterns(
    hours: int = Query(default=24, description="Hours of events to analyze"),
    event_service: EventProcessingService = Depends(get_event_service),
) -> SuccessResponse[list[dict[str, Any]]]:
    """Detect patterns in recent events."""
    try:
        # Get recent events
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)

        filter_criteria = EventFilter(
            event_time_start=start_time,
            event_time_end=end_time,
            limit=10000,
        )

        events = await event_service.query_events(filter_criteria)

        # Detect patterns
        detected_patterns = await event_service.detect_event_patterns(events)

        # Format response
        pattern_results = []
        for pattern, matching_events in detected_patterns:
            result = {
                "pattern": pattern,
                "matching_events": matching_events,
                "confidence": pattern.confidence,
                "match_count": len(matching_events),
                "time_range": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                },
            }
            pattern_results.append(result)

        return SuccessResponse(
            data=pattern_results,
            message=f"Detected {len(pattern_results)} patterns in {len(events)} events",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to detect patterns: {str(e)}"
        )


@router.get(
    "/real-time",
    response_model=SuccessResponse[list[AnomalyEvent]],
    summary="Get Real-time Events",
    description="""
    Get the most recent events for real-time monitoring.
    
    This endpoint is optimized for:
    - Dashboard real-time displays
    - Monitoring applications
    - Alert systems
    - Live event feeds
    
    **Features:**
    - Returns only recent events (last 5 minutes by default)
    - Filters for actionable events (high severity, unresolved)
    - Sorted by most recent first
    - Lightweight response format
    """,
)
async def get_realtime_events(
    minutes: int = Query(default=5, description="Minutes of recent events"),
    severity_filter: EventSeverity | None = Query(None, description="Minimum severity"),
    limit: int = Query(default=50, description="Maximum events to return"),
    event_service: EventProcessingService = Depends(get_event_service),
) -> SuccessResponse[list[AnomalyEvent]]:
    """Get real-time events for monitoring."""
    try:
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=minutes)

        # Build filter for real-time events
        severities = None
        if severity_filter:
            # Include specified severity and higher
            all_severities = [
                EventSeverity.INFO,
                EventSeverity.LOW,
                EventSeverity.MEDIUM,
                EventSeverity.HIGH,
                EventSeverity.CRITICAL,
            ]
            severity_index = all_severities.index(severity_filter)
            severities = all_severities[severity_index:]

        filter_criteria = EventFilter(
            event_time_start=start_time,
            event_time_end=end_time,
            severities=severities,
            statuses=[
                EventStatus.PENDING,
                EventStatus.PROCESSING,
                EventStatus.ACKNOWLEDGED,
            ],
            limit=limit,
            sort_by="event_time",
            sort_order="desc",
        )

        events = await event_service.query_events(filter_criteria)

        return SuccessResponse(
            data=events, message=f"Retrieved {len(events)} real-time events"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get real-time events: {str(e)}"
        )
