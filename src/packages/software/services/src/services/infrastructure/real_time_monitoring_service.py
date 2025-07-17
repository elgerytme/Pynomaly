"""Real-time monitoring and analytics service for comprehensive system observability.

This service provides real-time monitoring, user analytics, error tracking, and
performance monitoring capabilities for the comprehensive monitoring dashboard.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

from ...domain.models.monitoring import (
    Alert,
    AlertRule,
    AlertSeverity,
    FeatureUsageMetrics,
    SystemPerformanceSnapshot,
    UserEvent,
    UserEventType,
    UserSession,
    WebVitalMetric,
    WebVitalType,
)
from ...infrastructure.config.feature_flags import require_feature


class RealTimeMonitoringService:
    """Service for real-time monitoring and analytics dashboard."""

    def __init__(
        self,
        max_events_buffer: int = 10000,
        max_sessions_buffer: int = 1000,
        measurements_retention_hours: int = 24,
        enable_privacy_mode: bool = True,
    ):
        """Initialize real-time monitoring service.

        Args:
            max_events_buffer: Maximum events to keep in memory
            max_sessions_buffer: Maximum sessions to keep in memory
            measurements_retention_hours: Hours to retain measurements
            enable_privacy_mode: Enable privacy compliance features
        """
        self.max_events_buffer = max_events_buffer
        self.max_sessions_buffer = max_sessions_buffer
        self.measurements_retention_hours = measurements_retention_hours
        self.enable_privacy_mode = enable_privacy_mode

        # In-memory storage for real-time data
        self.active_sessions: dict[str, UserSession] = {}
        self.events_buffer: deque = deque(maxlen=max_events_buffer)
        self.web_vitals_buffer: deque = deque(maxlen=max_events_buffer)
        self.performance_snapshots: deque = deque(maxlen=1000)

        # Real-time measurements
        self.current_measurements = {
            "active_users": 0,
            "active_sessions": 0,
            "requests_per_second": 0.0,
            "error_rate": 0.0,
            "avg_response_time": 0.0,
        }

        # Error tracking
        self.error_counts: dict[str, int] = defaultdict(int)
        self.error_categories: dict[str, list[dict]] = defaultdict(list)
        self.recent_errors: deque = deque(maxlen=100)

        # Feature usage tracking
        self.feature_usage: dict[str, FeatureUsageMetrics] = {}

        # Alert management
        self.active_alerts: dict[UUID, Alert] = {}
        self.alert_rules: dict[UUID, AlertRule] = {}
        self.alert_callbacks: list[Callable] = []

        # Real-time subscribers (WebSocket connections)
        self.subscribers: set[Any] = set()  # WebSocket connections

        # Performance tracking
        self.request_times: deque = deque(maxlen=1000)  # Last 1000 requests
        self.error_times: deque = deque(maxlen=1000)  # Last 1000 errors

        # Background tasks
        self._background_tasks: list[asyncio.Task] = []
        self._is_running = False

    async def start_monitoring(self) -> None:
        """Start the real-time monitoring service."""
        if self._is_running:
            return

        self._is_running = True

        # Start background tasks
        self._background_tasks = [
            asyncio.create_task(self._performance_snapshot_task()),
            asyncio.create_task(self._session_cleanup_task()),
            asyncio.create_task(self._alert_evaluation_task()),
            asyncio.create_task(self._measurements_cleanup_task()),
        ]

    async def stop_monitoring(self) -> None:
        """Stop the real-time monitoring service."""
        self._is_running = False

        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()

        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        self._background_tasks.clear()

    @require_feature("user_analytics")
    async def track_user_event(
        self,
        session_id: str,
        event_type: UserEventType,
        event_name: str,
        user_id: UUID | None = None,
        properties: dict[str, Any] | None = None,
        page_url: str = "",
        response_time_ms: int | None = None,
        error_message: str | None = None,
    ) -> UserEvent:
        """Track a user event for analytics.

        Args:
            session_id: User session identifier
            event_type: Type of event
            event_name: Name of the event
            user_id: User ID (optional for anonymous tracking)
            properties: Additional event properties
            page_url: URL where event occurred
            response_time_ms: Response time for the event
            error_message: Error message if applicable

        Returns:
            Created UserEvent
        """
        event = UserEvent(
            event_id=uuid4(),
            user_id=user_id,
            session_id=session_id,
            event_type=event_type,
            event_name=event_name,
            page_url=page_url,
            properties=properties or {},
            response_time_ms=response_time_ms,
            error_message=error_message,
        )

        # Apply privacy compliance
        if self.enable_privacy_mode:
            event.anonymize()

        # Add to buffer
        self.events_buffer.append(event)

        # Update session if it exists
        if session_id in self.active_sessions:
            self.active_sessions[session_id].add_event(event)

        # Track error if present
        if error_message:
            await self._track_error(error_message, event)

        # Update real-time measurements
        await self._update_real_time_measurements()

        # Notify subscribers
        await self._notify_subscribers("user_event", event.__dict__)

        return event

    @require_feature("user_analytics")
    async def start_user_session(
        self,
        session_id: str,
        user_id: UUID | None = None,
        ip_address: str = "",
        user_agent: str = "",
        referrer: str = "",
        landing_page: str = "",
    ) -> UserSession:
        """Start a new user session.

        Args:
            session_id: Unique session identifier
            user_id: User ID (optional for anonymous users)
            ip_address: User's IP address
            user_agent: User's browser/client info
            referrer: Referring URL
            landing_page: First page visited

        Returns:
            Created UserSession
        """
        session = UserSession(
            session_id=session_id,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            referrer=referrer,
            landing_page=landing_page,
        )

        # Apply privacy compliance
        if self.enable_privacy_mode:
            session.anonymize()

        # Add to active sessions
        self.active_sessions[session_id] = session

        # Clean up old sessions if buffer is full
        if len(self.active_sessions) > self.max_sessions_buffer:
            oldest_session_id = min(
                self.active_sessions.keys(),
                key=lambda sid: self.active_sessions[sid].start_time,
            )
            await self.end_user_session(oldest_session_id)

        # Update measurements
        await self._update_real_time_measurements()

        # Notify subscribers
        await self._notify_subscribers("session_start", session.__dict__)

        return session

    @require_feature("user_analytics")
    async def end_user_session(self, session_id: str) -> UserSession | None:
        """End a user session.

        Args:
            session_id: Session to end

        Returns:
            Ended UserSession if it existed
        """
        if session_id not in self.active_sessions:
            return None

        session = self.active_sessions[session_id]
        session.end_session()

        # Remove from active sessions
        del self.active_sessions[session_id]

        # Update measurements
        await self._update_real_time_measurements()

        # Notify subscribers
        await self._notify_subscribers("session_end", session.__dict__)

        return session

    @require_feature("performance_monitoring")
    async def track_web_vital(
        self,
        session_id: str,
        vital_type: WebVitalType,
        value: float,
        page_url: str = "",
        user_id: UUID | None = None,
        device_type: str = "",
        connection_type: str = "",
    ) -> WebVitalMetric:
        """Track Core Web Vitals metric.

        Args:
            session_id: User session ID
            vital_type: Type of Web Vital
            value: Metric value
            page_url: URL where metric was measured
            user_id: User ID (optional)
            device_type: Device type
            connection_type: Connection type

        Returns:
            Created WebVitalMetric
        """
        rating = WebVitalMetric.calculate_rating(vital_type, value)

        metric = WebVitalMetric(
            metric_id=uuid4(),
            session_id=session_id,
            user_id=user_id,
            vital_type=vital_type,
            value=value,
            rating=rating,
            page_url=page_url,
            device_type=device_type,
            connection_type=connection_type,
        )

        # Add to buffer
        self.web_vitals_buffer.append(metric)

        # Notify subscribers
        await self._notify_subscribers("web_vital", metric.__dict__)

        return metric

    @require_feature("performance_monitoring")
    async def record_api_request(
        self,
        endpoint: str,
        method: str,
        response_time_ms: int,
        status_code: int,
        user_id: UUID | None = None,
        session_id: str | None = None,
        error_message: str | None = None,
    ) -> None:
        """Record API request for performance monitoring.

        Args:
            endpoint: API endpoint
            method: HTTP method
            response_time_ms: Response time in milliseconds
            status_code: HTTP status code
            user_id: User ID (optional)
            session_id: Session ID (optional)
            error_message: Error message if request failed
        """
        # Record request timing
        self.request_times.append(
            {
                "timestamp": datetime.utcnow(),
                "response_time": response_time_ms,
                "endpoint": endpoint,
                "method": method,
                "status_code": status_code,
            }
        )

        # Track errors
        if status_code >= 400 or error_message:
            await self._track_api_error(
                endpoint, method, status_code, error_message, user_id, session_id
            )

        # Update measurements
        await self._update_real_time_measurements()

        # Check for alerts
        await self._check_performance_alerts(response_time_ms, status_code)

    async def create_performance_snapshot(self) -> SystemPerformanceSnapshot:
        """Create a real-time performance snapshot.

        Returns:
            SystemPerformanceSnapshot with current measurements
        """
        # Calculate API performance measurements
        recent_requests = [
            req
            for req in self.request_times
            if req["timestamp"] > datetime.utcnow() - timedelta(minutes=5)
        ]

        if recent_requests:
            response_times = [req["response_time"] for req in recent_requests]
            response_times.sort()

            p50 = response_times[len(response_times) // 2] if response_times else 0
            p95 = (
                response_times[int(len(response_times) * 0.95)] if response_times else 0
            )
            p99 = (
                response_times[int(len(response_times) * 0.99)] if response_times else 0
            )

            error_count = sum(1 for req in recent_requests if req["status_code"] >= 400)
            error_rate = (
                (error_count / len(recent_requests)) * 100 if recent_requests else 0
            )
            rps = len(recent_requests) / 5.0  # 5-minute window
        else:
            p50 = p95 = p99 = error_rate = rps = 0

        # Calculate Web Vitals averages
        recent_vitals = [
            vital
            for vital in self.web_vitals_buffer
            if vital.timestamp > datetime.utcnow() - timedelta(minutes=5)
        ]

        fcp_values = [
            v.value
            for v in recent_vitals
            if v.vital_type == WebVitalType.FIRST_CONTENTFUL_PAINT
        ]
        lcp_values = [
            v.value
            for v in recent_vitals
            if v.vital_type == WebVitalType.LARGEST_CONTENTFUL_PAINT
        ]
        fid_values = [
            v.value
            for v in recent_vitals
            if v.vital_type == WebVitalType.FIRST_INPUT_DELAY
        ]
        cls_values = [
            v.value
            for v in recent_vitals
            if v.vital_type == WebVitalType.CUMULATIVE_LAYOUT_SHIFT
        ]

        snapshot = SystemPerformanceSnapshot(
            snapshot_id=uuid4(),
            api_response_time_p50=p50,
            api_response_time_p95=p95,
            api_response_time_p99=p99,
            requests_per_second=rps,
            error_rate_percentage=error_rate,
            active_users=len(
                set(s.user_id for s in self.active_sessions.values() if s.user_id)
            ),
            active_sessions=len(self.active_sessions),
            avg_fcp_ms=sum(fcp_values) / len(fcp_values) if fcp_values else 0,
            avg_lcp_ms=sum(lcp_values) / len(lcp_values) if lcp_values else 0,
            avg_fid_ms=sum(fid_values) / len(fid_values) if fid_values else 0,
            avg_cls_score=sum(cls_values) / len(cls_values) if cls_values else 0,
        )

        self.performance_snapshots.append(snapshot)

        # Notify subscribers
        await self._notify_subscribers(
            "performance_snapshot", snapshot.get_performance_summary()
        )

        return snapshot

    @require_feature("error_tracking")
    async def get_error_analytics(
        self, time_window: timedelta = timedelta(hours=1)
    ) -> dict[str, Any]:
        """Get error analytics for the dashboard.

        Args:
            time_window: Time window for analysis

        Returns:
            Error analytics data
        """
        cutoff_time = datetime.utcnow() - time_window

        # Get recent errors
        recent_errors = [
            error for error in self.recent_errors if error["timestamp"] > cutoff_time
        ]

        # Categorize errors
        error_by_type = defaultdict(int)
        error_by_endpoint = defaultdict(int)
        error_by_status_code = defaultdict(int)

        for error in recent_errors:
            error_by_type[error.get("type", "unknown")] += 1
            error_by_endpoint[error.get("endpoint", "unknown")] += 1
            error_by_status_code[error.get("status_code", 500)] += 1

        return {
            "time_window_hours": time_window.total_seconds() / 3600,
            "total_errors": len(recent_errors),
            "error_rate": self.current_measurements["error_rate"],
            "errors_by_type": dict(error_by_type),
            "errors_by_endpoint": dict(error_by_endpoint),
            "errors_by_status_code": dict(error_by_status_code),
            "recent_errors": list(self.recent_errors)[-10:],  # Last 10 errors
            "error_trends": await self._calculate_error_trends(time_window),
        }

    @require_feature("user_analytics")
    async def get_user_analytics(
        self, time_window: timedelta = timedelta(hours=1)
    ) -> dict[str, Any]:
        """Get user analytics for the dashboard.

        Args:
            time_window: Time window for analysis

        Returns:
            User analytics data
        """
        cutoff_time = datetime.utcnow() - time_window

        # Get recent events
        recent_events = [
            event for event in self.events_buffer if event.timestamp > cutoff_time
        ]

        # Calculate measurements
        unique_users = len(
            set(event.user_id for event in recent_events if event.user_id)
        )
        unique_sessions = len(set(event.session_id for event in recent_events))

        # Feature usage
        feature_usage = defaultdict(int)
        for event in recent_events:
            if event.event_type in [
                UserEventType.MODEL_PREDICTION,
                UserEventType.DATA_UPLOAD,
                UserEventType.DASHBOARD_ACCESS,
                UserEventType.REPORT_GENERATION,
            ]:
                feature_usage[event.event_type.value] += 1

        # Device breakdown
        device_usage = defaultdict(int)
        for session in self.active_sessions.values():
            device_usage[session.device_type or "unknown"] += 1

        return {
            "time_window_hours": time_window.total_seconds() / 3600,
            "total_events": len(recent_events),
            "unique_users": unique_users,
            "unique_sessions": unique_sessions,
            "active_sessions": len(self.active_sessions),
            "feature_usage": dict(feature_usage),
            "device_usage": dict(device_usage),
            "avg_session_duration": await self._calculate_avg_session_duration(),
        }

    async def get_real_time_dashboard_data(self) -> dict[str, Any]:
        """Get comprehensive real-time dashboard data.

        Returns:
            Complete dashboard data for real-time updates
        """
        latest_snapshot = (
            self.performance_snapshots[-1] if self.performance_snapshots else None
        )

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "system_health": {
                "health_score": latest_snapshot.get_health_score()
                if latest_snapshot
                else 0,
                "status": "healthy"
                if (latest_snapshot and latest_snapshot.get_health_score() > 80)
                else "degraded",
                "active_alerts": len(self.active_alerts),
                "critical_alerts": len(
                    [
                        a
                        for a in self.active_alerts.values()
                        if a.severity == AlertSeverity.CRITICAL
                    ]
                ),
            },
            "performance": latest_snapshot.get_performance_summary()
            if latest_snapshot
            else {},
            "users": {
                "active_users": self.current_measurements["active_users"],
                "active_sessions": self.current_measurements["active_sessions"],
                "total_events_last_hour": len(
                    [
                        e
                        for e in self.events_buffer
                        if e.timestamp > datetime.utcnow() - timedelta(hours=1)
                    ]
                ),
            },
            "api": {
                "requests_per_second": self.current_measurements["requests_per_second"],
                "avg_response_time": self.current_measurements["avg_response_time"],
                "error_rate": self.current_measurements["error_rate"],
            },
            "errors": await self.get_error_analytics(timedelta(hours=1)),
            "alerts": [alert.__dict__ for alert in self.active_alerts.values()],
        }

    async def add_subscriber(self, subscriber: Any) -> None:
        """Add real-time data subscriber (WebSocket connection).

        Args:
            subscriber: WebSocket connection or callback
        """
        self.subscribers.add(subscriber)

    async def remove_subscriber(self, subscriber: Any) -> None:
        """Remove real-time data subscriber.

        Args:
            subscriber: WebSocket connection or callback to remove
        """
        self.subscribers.discard(subscriber)

    async def _update_real_time_measurements(self) -> None:
        """Update real-time measurements."""
        # Update active users and sessions
        self.current_measurements["active_users"] = len(
            set(s.user_id for s in self.active_sessions.values() if s.user_id)
        )
        self.current_measurements["active_sessions"] = len(self.active_sessions)

        # Calculate requests per second (last 60 seconds)
        recent_requests = [
            req
            for req in self.request_times
            if req["timestamp"] > datetime.utcnow() - timedelta(seconds=60)
        ]
        self.current_measurements["requests_per_second"] = len(recent_requests) / 60.0

        # Calculate average response time (last 100 requests)
        if recent_requests:
            self.current_measurements["avg_response_time"] = sum(
                req["response_time"] for req in recent_requests[-100:]
            ) / min(len(recent_requests), 100)

            # Calculate error rate
            error_count = sum(1 for req in recent_requests if req["status_code"] >= 400)
            self.current_measurements["error_rate"] = (
                error_count / len(recent_requests)
            ) * 100
        else:
            self.current_measurements["avg_response_time"] = 0
            self.current_measurements["error_rate"] = 0

    async def _track_error(self, error_message: str, event: UserEvent) -> None:
        """Track an error for analytics."""
        error_data = {
            "timestamp": datetime.utcnow(),
            "message": error_message,
            "type": "user_event_error",
            "event_type": event.event_type.value,
            "page_url": event.page_url,
            "user_id": str(event.user_id) if event.user_id else None,
            "session_id": event.session_id,
        }

        self.recent_errors.append(error_data)
        self.error_counts[error_message] += 1

    async def _track_api_error(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        error_message: str | None,
        user_id: UUID | None,
        session_id: str | None,
    ) -> None:
        """Track an API error."""
        error_data = {
            "timestamp": datetime.utcnow(),
            "type": "api_error",
            "endpoint": endpoint,
            "method": method,
            "status_code": status_code,
            "message": error_message,
            "user_id": str(user_id) if user_id else None,
            "session_id": session_id,
        }

        self.recent_errors.append(error_data)
        self.error_counts[f"{method} {endpoint}"] += 1

    async def _notify_subscribers(self, event_type: str, data: Any) -> None:
        """Notify all subscribers of real-time events."""
        if not self.subscribers:
            return

        message = {
            "type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data,
        }

        # Remove failed subscribers
        failed_subscribers = set()

        for subscriber in self.subscribers:
            try:
                if hasattr(subscriber, "send_json"):
                    await subscriber.send_json(message)
                elif callable(subscriber):
                    await subscriber(message)
            except Exception:
                failed_subscribers.add(subscriber)

        # Clean up failed subscribers
        self.subscribers -= failed_subscribers

    async def _performance_snapshot_task(self) -> None:
        """Background task to create periodic performance snapshots."""
        while self._is_running:
            try:
                await self.create_performance_snapshot()
                await asyncio.sleep(30)  # Create snapshot every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(30)  # Continue on error

    async def _session_cleanup_task(self) -> None:
        """Background task to clean up inactive sessions."""
        while self._is_running:
            try:
                cutoff_time = datetime.utcnow() - timedelta(hours=1)
                inactive_sessions = [
                    session_id
                    for session_id, session in self.active_sessions.items()
                    if session.start_time < cutoff_time
                ]

                for session_id in inactive_sessions:
                    await self.end_user_session(session_id)

                await asyncio.sleep(300)  # Check every 5 minutes
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(300)

    async def _alert_evaluation_task(self) -> None:
        """Background task to evaluate alert rules."""
        while self._is_running:
            try:
                # This would evaluate alert rules against current measurements
                # Implementation depends on alert rule configuration
                await asyncio.sleep(60)  # Check every minute
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(60)

    async def _measurements_cleanup_task(self) -> None:
        """Background task to clean up old measurements."""
        while self._is_running:
            try:
                cutoff_time = datetime.utcnow() - timedelta(
                    hours=self.measurements_retention_hours
                )

                # Clean up old events
                self.events_buffer = deque(
                    [e for e in self.events_buffer if e.timestamp > cutoff_time],
                    maxlen=self.max_events_buffer,
                )

                # Clean up old web vitals
                self.web_vitals_buffer = deque(
                    [v for v in self.web_vitals_buffer if v.timestamp > cutoff_time],
                    maxlen=self.max_events_buffer,
                )

                # Clean up old request times
                self.request_times = deque(
                    [r for r in self.request_times if r["timestamp"] > cutoff_time],
                    maxlen=1000,
                )

                await asyncio.sleep(3600)  # Clean up every hour
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(3600)

    async def _calculate_error_trends(self, time_window: timedelta) -> dict[str, Any]:
        """Calculate error trends over time."""
        # Simplified trend calculation
        cutoff_time = datetime.utcnow() - time_window
        recent_errors = [
            error for error in self.recent_errors if error["timestamp"] > cutoff_time
        ]

        # Group errors by hour
        hourly_errors = defaultdict(int)
        for error in recent_errors:
            hour = error["timestamp"].replace(minute=0, second=0, microsecond=0)
            hourly_errors[hour] += 1

        return {
            "hourly_counts": {
                hour.isoformat(): count for hour, count in hourly_errors.items()
            },
            "trend": "increasing"
            if len(recent_errors) > len(self.recent_errors) // 2
            else "stable",
        }

    async def _calculate_avg_session_duration(self) -> float:
        """Calculate average session duration in seconds."""
        ended_sessions = [
            s for s in self.active_sessions.values() if s.duration_seconds is not None
        ]

        if not ended_sessions:
            return 0.0

        return sum(s.duration_seconds for s in ended_sessions) / len(ended_sessions)

    async def _check_performance_alerts(
        self, response_time_ms: int, status_code: int
    ) -> None:
        """Check if performance measurements trigger alerts."""
        # This would check against configured alert rules
        # Implementation depends on alert rule configuration
        pass
