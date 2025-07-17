"""Advanced error tracking and categorization service.

This service provides comprehensive error tracking, categorization, analysis,
and alerting capabilities for production monitoring.
"""

from __future__ import annotations

import re
import traceback
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from ...domain.models.monitoring import Alert, AlertRule, AlertSeverity
from ...infrastructure.config.feature_flags import require_feature


class ErrorCategory(Enum):
    """Categories for error classification."""

    # Application errors
    VALIDATION_ERROR = "validation_error"
    BUSINESS_LOGIC_ERROR = "business_logic_error"
    AUTHENTICATION_ERROR = "authentication_error"
    AUTHORIZATION_ERROR = "authorization_error"

    # Infrastructure errors
    DATABASE_ERROR = "database_error"
    NETWORK_ERROR = "network_error"
    EXTERNAL_SERVICE_ERROR = "external_service_error"
    CACHE_ERROR = "cache_error"

    # System errors
    MEMORY_ERROR = "memory_error"
    TIMEOUT_ERROR = "timeout_error"
    CONFIGURATION_ERROR = "configuration_error"
    DEPENDENCY_ERROR = "dependency_error"

    # Client errors
    CLIENT_ERROR = "client_error"
    BAD_REQUEST = "bad_request"
    NOT_FOUND = "not_found"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"

    # Unknown/uncategorized
    UNKNOWN_ERROR = "unknown_error"


class ErrorSeverity(Enum):
    """Error severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorPattern:
    """Pattern for error classification."""

    def __init__(
        self,
        pattern_id: str,
        name: str,
        regex_pattern: str,
        category: ErrorCategory,
        severity: ErrorSeverity,
        description: str = "",
        suggested_action: str = "",
    ):
        self.pattern_id = pattern_id
        self.name = name
        self.regex_pattern = re.compile(regex_pattern, re.IGNORECASE)
        self.category = category
        self.severity = severity
        self.description = description
        self.suggested_action = suggested_action

    def matches(self, error_message: str) -> bool:
        """Check if error message matches this pattern."""
        return bool(self.regex_pattern.search(error_message))


class ErrorInstance:
    """Individual error instance."""

    def __init__(
        self,
        error_id: UUID,
        message: str,
        category: ErrorCategory,
        severity: ErrorSeverity,
        timestamp: datetime,
        stack_trace: str | None = None,
        context: dict[str, Any] | None = None,
        user_id: UUID | None = None,
        session_id: str | None = None,
        endpoint: str | None = None,
        request_id: str | None = None,
    ):
        self.error_id = error_id
        self.message = message
        self.category = category
        self.severity = severity
        self.timestamp = timestamp
        self.stack_trace = stack_trace
        self.context = context or {}
        self.user_id = user_id
        self.session_id = session_id
        self.endpoint = endpoint
        self.request_id = request_id
        self.fingerprint = self._calculate_fingerprint()
        self.is_resolved = False
        self.resolution_note: str | None = None
        self.resolved_at: datetime | None = None
        self.resolved_by: UUID | None = None

    def _calculate_fingerprint(self) -> str:
        """Calculate unique fingerprint for error grouping."""
        import hashlib

        # Use message + category + first few lines of stack trace
        fingerprint_data = f"{self.message}:{self.category.value}"

        if self.stack_trace:
            # Use first 3 lines of stack trace for fingerprinting
            stack_lines = self.stack_trace.split("\n")[:3]
            fingerprint_data += ":" + ":".join(stack_lines)

        return hashlib.md5(fingerprint_data.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "error_id": str(self.error_id),
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
            "stack_trace": self.stack_trace,
            "context": self.context,
            "user_id": str(self.user_id) if self.user_id else None,
            "session_id": self.session_id,
            "endpoint": self.endpoint,
            "request_id": self.request_id,
            "fingerprint": self.fingerprint,
            "is_resolved": self.is_resolved,
            "resolution_note": self.resolution_note,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolved_by": str(self.resolved_by) if self.resolved_by else None,
        }


class ErrorGroup:
    """Group of similar errors."""

    def __init__(self, fingerprint: str, first_error: ErrorInstance):
        self.fingerprint = fingerprint
        self.first_seen = first_error.timestamp
        self.last_seen = first_error.timestamp
        self.count = 1
        self.category = first_error.category
        self.severity = first_error.severity
        self.message_template = first_error.message
        self.errors: list[ErrorInstance] = [first_error]
        self.is_silenced = False
        self.silenced_until: datetime | None = None
        self.tags: set[str] = set()
        self.assigned_to: UUID | None = None
        self.status = "new"  # new, investigating, resolved, ignored

    def add_error(self, error: ErrorInstance) -> None:
        """Add error to this group."""
        self.errors.append(error)
        self.count += 1
        self.last_seen = max(self.last_seen, error.timestamp)

        # Update severity if new error is more severe
        severity_order = [
            ErrorSeverity.LOW,
            ErrorSeverity.MEDIUM,
            ErrorSeverity.HIGH,
            ErrorSeverity.CRITICAL,
        ]
        if severity_order.index(error.severity) > severity_order.index(self.severity):
            self.severity = error.severity

    def get_frequency(self, time_window: timedelta) -> float:
        """Get error frequency per hour in the given time window."""
        cutoff_time = datetime.utcnow() - time_window
        recent_errors = [e for e in self.errors if e.timestamp > cutoff_time]
        return len(recent_errors) / (time_window.total_seconds() / 3600)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "fingerprint": self.fingerprint,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "count": self.count,
            "category": self.category.value,
            "severity": self.severity.value,
            "message_template": self.message_template,
            "is_silenced": self.is_silenced,
            "silenced_until": self.silenced_until.isoformat()
            if self.silenced_until
            else None,
            "tags": list(self.tags),
            "assigned_to": str(self.assigned_to) if self.assigned_to else None,
            "status": self.status,
            "recent_errors": [e.to_dict() for e in self.errors[-5:]],  # Last 5 errors
        }


class ErrorTrackingService:
    """Service for advanced error tracking and analysis."""

    def __init__(self, max_errors_buffer: int = 10000, retention_days: int = 30):
        """Initialize error tracking service.

        Args:
            max_errors_buffer: Maximum errors to keep in memory
            retention_days: Days to retain error data
        """
        self.max_errors_buffer = max_errors_buffer
        self.retention_days = retention_days

        # Error storage
        self.error_groups: dict[str, ErrorGroup] = {}
        self.recent_errors: deque = deque(maxlen=max_errors_buffer)
        self.error_patterns: list[ErrorPattern] = []

        # Statistics
        self.error_counts_by_category: dict[ErrorCategory, int] = defaultdict(int)
        self.error_counts_by_severity: dict[ErrorSeverity, int] = defaultdict(int)
        self.hourly_error_counts: dict[datetime, int] = defaultdict(int)

        # Alerting
        self.alert_rules: list[AlertRule] = []
        self.active_alerts: dict[UUID, Alert] = {}

        # Initialize default error patterns
        self._initialize_default_patterns()

    @require_feature("error_tracking")
    def track_error(
        self,
        message: str,
        exception: Exception | None = None,
        stack_trace: str | None = None,
        context: dict[str, Any] | None = None,
        user_id: UUID | None = None,
        session_id: str | None = None,
        endpoint: str | None = None,
        request_id: str | None = None,
    ) -> ErrorInstance:
        """Track a new error.

        Args:
            message: Error message
            exception: Exception object (optional)
            stack_trace: Stack trace string (optional)
            context: Additional context data
            user_id: User ID (optional)
            session_id: Session ID (optional)
            endpoint: API endpoint where error occurred
            request_id: Request ID for correlation

        Returns:
            Created ErrorInstance
        """
        # Extract stack trace from exception if not provided
        if exception and not stack_trace:
            stack_trace = "".join(
                traceback.format_exception(
                    type(exception), exception, exception.__traceback__
                )
            )

        # Categorize error
        category, severity = self._categorize_error(message, exception)

        # Create error instance
        error = ErrorInstance(
            error_id=uuid4(),
            message=message,
            category=category,
            severity=severity,
            timestamp=datetime.utcnow(),
            stack_trace=stack_trace,
            context=context,
            user_id=user_id,
            session_id=session_id,
            endpoint=endpoint,
            request_id=request_id,
        )

        # Add to recent errors
        self.recent_errors.append(error)

        # Group error
        self._group_error(error)

        # Update statistics
        self._update_statistics(error)

        # Check for alerts
        self._check_error_alerts(error)

        return error

    @require_feature("error_tracking")
    def get_error_dashboard_data(
        self, time_window: timedelta = timedelta(hours=24)
    ) -> dict[str, Any]:
        """Get error dashboard data.

        Args:
            time_window: Time window for analysis

        Returns:
            Dashboard data including error trends, top errors, etc.
        """
        cutoff_time = datetime.utcnow() - time_window

        # Get recent errors
        recent_errors = [
            error for error in self.recent_errors if error.timestamp > cutoff_time
        ]

        # Calculate measurements
        total_errors = len(recent_errors)
        error_rate = total_errors / (
            time_window.total_seconds() / 3600
        )  # errors per hour

        # Group by category
        category_counts = defaultdict(int)
        severity_counts = defaultdict(int)

        for error in recent_errors:
            category_counts[error.category.value] += 1
            severity_counts[error.severity.value] += 1

        # Get top error groups
        top_error_groups = sorted(
            [
                group
                for group in self.error_groups.values()
                if group.last_seen > cutoff_time
            ],
            key=lambda g: g.count,
            reverse=True,
        )[:10]

        # Calculate trends
        error_trends = self._calculate_error_trends(time_window)

        return {
            "time_window_hours": time_window.total_seconds() / 3600,
            "total_errors": total_errors,
            "error_rate_per_hour": error_rate,
            "errors_by_category": dict(category_counts),
            "errors_by_severity": dict(severity_counts),
            "top_error_groups": [group.to_dict() for group in top_error_groups],
            "error_trends": error_trends,
            "active_alerts": len(self.active_alerts),
            "resolved_errors": len(
                [
                    group
                    for group in self.error_groups.values()
                    if group.status == "resolved"
                ]
            ),
            "recent_errors": [
                error.to_dict() for error in list(self.recent_errors)[-10:]
            ],
        }

    @require_feature("error_tracking")
    def get_error_group(self, fingerprint: str) -> ErrorGroup | None:
        """Get error group by fingerprint.

        Args:
            fingerprint: Error group fingerprint

        Returns:
            ErrorGroup if found, None otherwise
        """
        return self.error_groups.get(fingerprint)

    @require_feature("error_tracking")
    def resolve_error_group(
        self, fingerprint: str, resolved_by: UUID, resolution_note: str = ""
    ) -> bool:
        """Mark error group as resolved.

        Args:
            fingerprint: Error group fingerprint
            resolved_by: User ID who resolved the error
            resolution_note: Resolution note

        Returns:
            True if resolved successfully, False if group not found
        """
        if fingerprint not in self.error_groups:
            return False

        group = self.error_groups[fingerprint]
        group.status = "resolved"

        # Mark all errors in group as resolved
        for error in group.errors:
            error.is_resolved = True
            error.resolved_at = datetime.utcnow()
            error.resolved_by = resolved_by
            error.resolution_note = resolution_note

        return True

    @require_feature("error_tracking")
    def silence_error_group(self, fingerprint: str, duration: timedelta) -> bool:
        """Silence error group for a duration.

        Args:
            fingerprint: Error group fingerprint
            duration: Duration to silence

        Returns:
            True if silenced successfully, False if group not found
        """
        if fingerprint not in self.error_groups:
            return False

        group = self.error_groups[fingerprint]
        group.is_silenced = True
        group.silenced_until = datetime.utcnow() + duration

        return True

    @require_feature("error_tracking")
    def add_error_pattern(self, pattern: ErrorPattern) -> None:
        """Add custom error pattern for classification.

        Args:
            pattern: ErrorPattern to add
        """
        self.error_patterns.append(pattern)

    @require_feature("error_tracking")
    def get_error_analytics(
        self, time_window: timedelta = timedelta(days=7)
    ) -> dict[str, Any]:
        """Get comprehensive error analytics.

        Args:
            time_window: Time window for analysis

        Returns:
            Comprehensive error analytics
        """
        cutoff_time = datetime.utcnow() - time_window

        # Get recent errors
        recent_errors = [
            error for error in self.recent_errors if error.timestamp > cutoff_time
        ]

        if not recent_errors:
            return {"message": "No errors in the specified time window"}

        # Calculate detailed analytics
        analytics = {
            "summary": {
                "total_errors": len(recent_errors),
                "unique_error_groups": len(
                    [
                        group
                        for group in self.error_groups.values()
                        if group.last_seen > cutoff_time
                    ]
                ),
                "resolution_rate": self._calculate_resolution_rate(cutoff_time),
                "avg_time_to_resolution": self._calculate_avg_resolution_time(),
            },
            "trends": self._calculate_detailed_trends(recent_errors, time_window),
            "top_affected_endpoints": self._get_top_affected_endpoints(recent_errors),
            "user_impact": self._calculate_user_impact(recent_errors),
            "performance_impact": self._calculate_performance_impact(recent_errors),
            "recommendations": self._generate_recommendations(recent_errors),
        }

        return analytics

    def _initialize_default_patterns(self) -> None:
        """Initialize default error patterns."""
        patterns = [
            ErrorPattern(
                "validation_error",
                "Validation Error",
                r"validation|invalid|required|missing|empty",
                ErrorCategory.VALIDATION_ERROR,
                ErrorSeverity.LOW,
                "Input validation failed",
                "Check input validation rules and user input",
            ),
            ErrorPattern(
                "database_connection",
                "Database Connection Error",
                r"database|connection|sql|postgres|mysql|timeout",
                ErrorCategory.DATABASE_ERROR,
                ErrorSeverity.HIGH,
                "Database connection or query failed",
                "Check database connectivity and query performance",
            ),
            ErrorPattern(
                "authentication_error",
                "Authentication Error",
                r"authentication|unauthorized|login|token|session",
                ErrorCategory.AUTHENTICATION_ERROR,
                ErrorSeverity.MEDIUM,
                "User authentication failed",
                "Check authentication configuration and user credentials",
            ),
            ErrorPattern(
                "network_error",
                "Network Error",
                r"network|connection refused|timeout|unreachable",
                ErrorCategory.NETWORK_ERROR,
                ErrorSeverity.HIGH,
                "Network connectivity issue",
                "Check network configuration and external service availability",
            ),
            ErrorPattern(
                "memory_error",
                "Memory Error",
                r"memory|out of memory|oom|allocation",
                ErrorCategory.MEMORY_ERROR,
                ErrorSeverity.CRITICAL,
                "Memory allocation failed",
                "Check memory usage and optimize resource allocation",
            ),
            ErrorPattern(
                "rate_limit",
                "Rate Limit Error",
                r"rate limit|too many requests|throttle",
                ErrorCategory.RATE_LIMIT_EXCEEDED,
                ErrorSeverity.MEDIUM,
                "Rate limit exceeded",
                "Implement proper rate limiting and user guidance",
            ),
        ]

        self.error_patterns.extend(patterns)

    def _categorize_error(
        self, message: str, exception: Exception | None = None
    ) -> tuple[ErrorCategory, ErrorSeverity]:
        """Categorize error based on message and exception type."""
        # Check custom patterns first
        for pattern in self.error_patterns:
            if pattern.matches(message):
                return pattern.category, pattern.severity

        # Check exception type
        if exception:
            exception_name = type(exception).__name__.lower()

            if "validation" in exception_name or "value" in exception_name:
                return ErrorCategory.VALIDATION_ERROR, ErrorSeverity.LOW
            elif "connection" in exception_name or "timeout" in exception_name:
                return ErrorCategory.NETWORK_ERROR, ErrorSeverity.HIGH
            elif "permission" in exception_name or "forbidden" in exception_name:
                return ErrorCategory.AUTHORIZATION_ERROR, ErrorSeverity.MEDIUM
            elif "memory" in exception_name:
                return ErrorCategory.MEMORY_ERROR, ErrorSeverity.CRITICAL

        # Default categorization
        return ErrorCategory.UNKNOWN_ERROR, ErrorSeverity.MEDIUM

    def _group_error(self, error: ErrorInstance) -> None:
        """Group error by fingerprint."""
        fingerprint = error.fingerprint

        if fingerprint in self.error_groups:
            self.error_groups[fingerprint].add_error(error)
        else:
            self.error_groups[fingerprint] = ErrorGroup(fingerprint, error)

    def _update_statistics(self, error: ErrorInstance) -> None:
        """Update error statistics."""
        self.error_counts_by_category[error.category] += 1
        self.error_counts_by_severity[error.severity] += 1

        # Update hourly counts
        hour = error.timestamp.replace(minute=0, second=0, microsecond=0)
        self.hourly_error_counts[hour] += 1

    def _check_error_alerts(self, error: ErrorInstance) -> None:
        """Check if error triggers any alerts."""
        # This would implement alert rule evaluation
        # For now, just create a simple alert for critical errors
        if error.severity == ErrorSeverity.CRITICAL:
            alert = Alert(
                alert_id=uuid4(),
                rule_id=uuid4(),  # Would reference actual rule
                rule_name="Critical Error Alert",
                metric_name="error_severity",
                metric_value=error.severity.value,
                threshold="critical",
                severity=AlertSeverity.CRITICAL,
                message=f"Critical error occurred: {error.message}",
            )
            self.active_alerts[alert.alert_id] = alert

    def _calculate_error_trends(self, time_window: timedelta) -> dict[str, Any]:
        """Calculate error trends over time."""
        cutoff_time = datetime.utcnow() - time_window

        # Group errors by hour
        hourly_counts = defaultdict(int)
        for error in self.recent_errors:
            if error.timestamp > cutoff_time:
                hour = error.timestamp.replace(minute=0, second=0, microsecond=0)
                hourly_counts[hour] += 1

        # Calculate trend
        hours = sorted(hourly_counts.keys())
        if len(hours) >= 2:
            recent_avg = sum(hourly_counts[h] for h in hours[-3:]) / min(3, len(hours))
            earlier_avg = sum(hourly_counts[h] for h in hours[:-3]) / max(
                1, len(hours) - 3
            )
            trend = "increasing" if recent_avg > earlier_avg else "decreasing"
        else:
            trend = "stable"

        return {
            "hourly_counts": {
                hour.isoformat(): count for hour, count in hourly_counts.items()
            },
            "trend": trend,
        }

    def _calculate_resolution_rate(self, cutoff_time: datetime) -> float:
        """Calculate error resolution rate."""
        relevant_groups = [
            group
            for group in self.error_groups.values()
            if group.first_seen > cutoff_time
        ]

        if not relevant_groups:
            return 0.0

        resolved_groups = len(
            [group for group in relevant_groups if group.status == "resolved"]
        )

        return (resolved_groups / len(relevant_groups)) * 100

    def _calculate_avg_resolution_time(self) -> float:
        """Calculate average time to resolution in hours."""
        resolved_groups = [
            group for group in self.error_groups.values() if group.status == "resolved"
        ]

        if not resolved_groups:
            return 0.0

        resolution_times = []
        for group in resolved_groups:
            resolved_errors = [
                e for e in group.errors if e.is_resolved and e.resolved_at
            ]
            if resolved_errors:
                first_error = min(group.errors, key=lambda e: e.timestamp)
                last_resolved = max(resolved_errors, key=lambda e: e.resolved_at)
                resolution_time = (
                    last_resolved.resolved_at - first_error.timestamp
                ).total_seconds() / 3600
                resolution_times.append(resolution_time)

        return (
            sum(resolution_times) / len(resolution_times) if resolution_times else 0.0
        )

    def _calculate_detailed_trends(
        self, errors: list[ErrorInstance], time_window: timedelta
    ) -> dict[str, Any]:
        """Calculate detailed error trends."""
        # This would implement more sophisticated trend analysis
        return {"message": "Detailed trends calculation not yet implemented"}

    def _get_top_affected_endpoints(
        self, errors: list[ErrorInstance]
    ) -> list[dict[str, Any]]:
        """Get top affected API endpoints."""
        endpoint_counts = defaultdict(int)
        for error in errors:
            if error.endpoint:
                endpoint_counts[error.endpoint] += 1

        return [
            {"endpoint": endpoint, "error_count": count}
            for endpoint, count in sorted(
                endpoint_counts.items(), key=lambda x: x[1], reverse=True
            )[:10]
        ]

    def _calculate_user_impact(self, errors: list[ErrorInstance]) -> dict[str, Any]:
        """Calculate user impact measurements."""
        affected_users = set(error.user_id for error in errors if error.user_id)
        affected_sessions = set(
            error.session_id for error in errors if error.session_id
        )

        return {
            "affected_users": len(affected_users),
            "affected_sessions": len(affected_sessions),
            "errors_per_user": len(errors) / max(1, len(affected_users)),
        }

    def _calculate_performance_impact(
        self, errors: list[ErrorInstance]
    ) -> dict[str, Any]:
        """Calculate performance impact of errors."""
        # This would analyze how errors affect system performance
        return {"message": "Performance impact calculation not yet implemented"}

    def _generate_recommendations(self, errors: list[ErrorInstance]) -> list[str]:
        """Generate recommendations based on error patterns."""
        recommendations = []

        # Analyze error categories
        category_counts = defaultdict(int)
        for error in errors:
            category_counts[error.category] += 1

        # Generate category-specific recommendations
        if category_counts[ErrorCategory.VALIDATION_ERROR] > len(errors) * 0.3:
            recommendations.append(
                "High number of validation errors detected. Consider improving input validation and user experience."
            )

        if category_counts[ErrorCategory.DATABASE_ERROR] > 0:
            recommendations.append(
                "Database errors detected. Check database performance and connection pooling."
            )

        if category_counts[ErrorCategory.NETWORK_ERROR] > 0:
            recommendations.append(
                "Network errors detected. Review external service dependencies and timeouts."
            )

        return recommendations
