"""Production monitoring service for anomaly detection operations.

This module provides comprehensive production monitoring capabilities including
structured logging, error tracking, performance monitoring, and health checks.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
import traceback
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from uuid import uuid4

import structlog
import numpy as np
import pandas as pd
from structlog import get_logger

from pynomaly.domain.entities import Dataset, Detector, DetectionResult
from pynomaly.domain.exceptions import DomainError, InfrastructureError
from .health_service import HealthService, HealthStatus, SystemMetrics
from .performance_monitor import PerformanceMonitor, PerformanceMetrics, PerformanceAlert


class LogLevel(Enum):
    """Logging levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MonitoringType(Enum):
    """Types of monitoring operations."""
    HEALTH_CHECK = "health_check"
    PERFORMANCE = "performance"
    ERROR_TRACKING = "error_tracking"
    AUDIT = "audit"
    SECURITY = "security"
    BUSINESS = "business"


@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    level: LogLevel = LogLevel.INFO
    message: str = ""
    operation: str = ""
    component: str = ""
    request_id: str = field(default_factory=lambda: str(uuid4()))
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    duration_ms: Optional[float] = None
    status: str = "success"
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "message": self.message,
            "operation": self.operation,
            "component": self.component,
            "request_id": self.request_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "stack_trace": self.stack_trace,
            "metadata": self.metadata
        }


@dataclass
class ErrorReport:
    """Comprehensive error report."""
    error_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    error_type: str = ""
    error_message: str = ""
    component: str = ""
    operation: str = ""
    stack_trace: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    severity: str = "medium"
    resolved: bool = False
    resolution_notes: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "error_id": self.error_id,
            "timestamp": self.timestamp.isoformat(),
            "error_type": self.error_type,
            "error_message": self.error_message,
            "component": self.component,
            "operation": self.operation,
            "stack_trace": self.stack_trace,
            "context": self.context,
            "severity": self.severity,
            "resolved": self.resolved,
            "resolution_notes": self.resolution_notes
        }


@dataclass
class AuditEvent:
    """Security and operational audit event."""
    event_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    event_type: str = ""
    user_id: Optional[str] = None
    operation: str = ""
    resource: str = ""
    action: str = ""
    outcome: str = "success"
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "user_id": self.user_id,
            "operation": self.operation,
            "resource": self.resource,
            "action": self.action,
            "outcome": self.outcome,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "metadata": self.metadata
        }


class ProductionMonitor:
    """Comprehensive production monitoring service."""
    
    def __init__(
        self,
        log_level: LogLevel = LogLevel.INFO,
        log_file: Optional[Path] = None,
        enable_performance_monitoring: bool = True,
        enable_health_checks: bool = True,
        enable_error_tracking: bool = True,
        enable_audit_logging: bool = True,
        max_log_entries: int = 10000,
        max_error_reports: int = 1000,
        max_audit_events: int = 5000
    ):
        """Initialize production monitor.
        
        Args:
            log_level: Minimum log level to capture
            log_file: Optional file path for log output
            enable_performance_monitoring: Enable performance tracking
            enable_health_checks: Enable health monitoring
            enable_error_tracking: Enable error reporting
            enable_audit_logging: Enable audit trail
            max_log_entries: Maximum log entries to keep in memory
            max_error_reports: Maximum error reports to keep
            max_audit_events: Maximum audit events to keep
        """
        self.log_level = log_level
        self.log_file = log_file
        self.enable_performance_monitoring = enable_performance_monitoring
        self.enable_health_checks = enable_health_checks
        self.enable_error_tracking = enable_error_tracking
        self.enable_audit_logging = enable_audit_logging
        
        # Storage
        self.log_entries: List[LogEntry] = []
        self.error_reports: List[ErrorReport] = []
        self.audit_events: List[AuditEvent] = []
        self.max_log_entries = max_log_entries
        self.max_error_reports = max_error_reports
        self.max_audit_events = max_audit_events
        
        # Monitoring components
        self.health_service = HealthService() if enable_health_checks else None
        self.performance_monitor = PerformanceMonitor() if enable_performance_monitoring else None
        
        # Initialize structured logging
        self._setup_logging()
        
        # Logger instance
        self.logger = get_logger(__name__)
    
    def _setup_logging(self) -> None:
        """Setup structured logging configuration."""
        processors = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="ISO"),
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.JSONRenderer()
        ]
        
        # Configure structlog
        structlog.configure(
            processors=processors,
            wrapper_class=structlog.make_filtering_bound_logger(
                getattr(logging, self.log_level.value.upper())
            ),
            context_class=dict,
            logger_factory=structlog.WriteLoggerFactory(),
            cache_logger_on_first_use=True,
        )
        
        # Setup file logging if specified
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setFormatter(logging.Formatter('%(message)s'))
            
            # Get root logger and add handler
            root_logger = logging.getLogger()
            root_logger.addHandler(file_handler)
            root_logger.setLevel(getattr(logging, self.log_level.value.upper()))
    
    def log(
        self,
        level: LogLevel,
        message: str,
        operation: str = "",
        component: str = "",
        request_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        duration_ms: Optional[float] = None,
        status: str = "success",
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
        **metadata
    ) -> None:
        """Log a structured entry.
        
        Args:
            level: Log level
            message: Log message
            operation: Operation being performed
            component: Component generating the log
            request_id: Request identifier
            user_id: User identifier
            session_id: Session identifier
            duration_ms: Operation duration in milliseconds
            status: Operation status
            error_type: Error type if applicable
            error_message: Error message if applicable
            **metadata: Additional metadata
        """
        if level.value < self.log_level.value:
            return
        
        # Create log entry
        log_entry = LogEntry(
            level=level,
            message=message,
            operation=operation,
            component=component,
            request_id=request_id or str(uuid4()),
            user_id=user_id,
            session_id=session_id,
            duration_ms=duration_ms,
            status=status,
            error_type=error_type,
            error_message=error_message,
            metadata=metadata
        )
        
        # Store in memory
        self.log_entries.append(log_entry)
        if len(self.log_entries) > self.max_log_entries:
            self.log_entries = self.log_entries[-self.max_log_entries:]
        
        # Log using structlog
        log_data = log_entry.to_dict()
        getattr(self.logger, level.value)(message, **log_data)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self.log(LogLevel.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self.log(LogLevel.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self.log(LogLevel.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs) -> None:
        """Log critical message."""
        self.log(LogLevel.CRITICAL, message, **kwargs)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self.log(LogLevel.DEBUG, message, **kwargs)
    
    def report_error(
        self,
        error: Exception,
        component: str,
        operation: str,
        context: Optional[Dict[str, Any]] = None,
        severity: str = "medium"
    ) -> str:
        """Report and track an error.
        
        Args:
            error: Exception that occurred
            component: Component where error occurred
            operation: Operation being performed
            context: Additional context
            severity: Error severity (low, medium, high, critical)
            
        Returns:
            Error ID for tracking
        """
        if not self.enable_error_tracking:
            return ""
        
        error_report = ErrorReport(
            error_type=type(error).__name__,
            error_message=str(error),
            component=component,
            operation=operation,
            stack_trace=traceback.format_exc(),
            context=context or {},
            severity=severity
        )
        
        # Store error report
        self.error_reports.append(error_report)
        if len(self.error_reports) > self.max_error_reports:
            self.error_reports = self.error_reports[-self.max_error_reports:]
        
        # Log the error
        self.error(
            f"Error in {component}.{operation}: {error}",
            component=component,
            operation=operation,
            error_type=error_report.error_type,
            error_message=error_report.error_message,
            error_id=error_report.error_id,
            severity=severity
        )
        
        return error_report.error_id
    
    def audit_event(
        self,
        event_type: str,
        operation: str,
        resource: str,
        action: str,
        user_id: Optional[str] = None,
        outcome: str = "success",
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        **metadata
    ) -> str:
        """Record an audit event.
        
        Args:
            event_type: Type of event (access, modification, etc.)
            operation: Operation performed
            resource: Resource affected
            action: Action taken
            user_id: User performing action
            outcome: Result of action
            ip_address: Client IP address
            user_agent: Client user agent
            **metadata: Additional metadata
            
        Returns:
            Event ID for tracking
        """
        if not self.enable_audit_logging:
            return ""
        
        audit_event = AuditEvent(
            event_type=event_type,
            user_id=user_id,
            operation=operation,
            resource=resource,
            action=action,
            outcome=outcome,
            ip_address=ip_address,
            user_agent=user_agent,
            metadata=metadata
        )
        
        # Store audit event
        self.audit_events.append(audit_event)
        if len(self.audit_events) > self.max_audit_events:
            self.audit_events = self.audit_events[-self.max_audit_events:]
        
        # Log the audit event
        self.info(
            f"Audit: {event_type} - {operation} on {resource}",
            operation=operation,
            component="audit",
            event_type=event_type,
            resource=resource,
            action=action,
            user_id=user_id,
            outcome=outcome,
            event_id=audit_event.event_id
        )
        
        return audit_event.event_id
    
    @contextmanager
    def monitor_operation(
        self,
        operation: str,
        component: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        **metadata
    ):
        """Context manager for monitoring operations.
        
        Args:
            operation: Operation name
            component: Component performing operation
            user_id: User identifier
            session_id: Session identifier
            **metadata: Additional metadata
        """
        request_id = str(uuid4())
        start_time = time.time()
        
        self.info(
            f"Starting {operation}",
            operation=operation,
            component=component,
            request_id=request_id,
            user_id=user_id,
            session_id=session_id,
            **metadata
        )
        
        try:
            yield request_id
            
            duration_ms = (time.time() - start_time) * 1000
            self.info(
                f"Completed {operation}",
                operation=operation,
                component=component,
                request_id=request_id,
                user_id=user_id,
                session_id=session_id,
                duration_ms=duration_ms,
                status="success",
                **metadata
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            error_id = self.report_error(e, component, operation)
            
            self.error(
                f"Failed {operation}: {e}",
                operation=operation,
                component=component,
                request_id=request_id,
                user_id=user_id,
                session_id=session_id,
                duration_ms=duration_ms,
                status="error",
                error_id=error_id,
                **metadata
            )
            raise
    
    @asynccontextmanager
    async def monitor_async_operation(
        self,
        operation: str,
        component: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        **metadata
    ):
        """Async context manager for monitoring operations.
        
        Args:
            operation: Operation name
            component: Component performing operation
            user_id: User identifier
            session_id: Session identifier
            **metadata: Additional metadata
        """
        request_id = str(uuid4())
        start_time = time.time()
        
        self.info(
            f"Starting async {operation}",
            operation=operation,
            component=component,
            request_id=request_id,
            user_id=user_id,
            session_id=session_id,
            **metadata
        )
        
        try:
            yield request_id
            
            duration_ms = (time.time() - start_time) * 1000
            self.info(
                f"Completed async {operation}",
                operation=operation,
                component=component,
                request_id=request_id,
                user_id=user_id,
                session_id=session_id,
                duration_ms=duration_ms,
                status="success",
                **metadata
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            error_id = self.report_error(e, component, operation)
            
            self.error(
                f"Failed async {operation}: {e}",
                operation=operation,
                component=component,
                request_id=request_id,
                user_id=user_id,
                session_id=session_id,
                duration_ms=duration_ms,
                status="error",
                error_id=error_id,
                **metadata
            )
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check.
        
        Returns:
            Health check results
        """
        if not self.health_service:
            return {"status": "disabled", "message": "Health checks disabled"}
        
        try:
            with self.monitor_operation("health_check", "monitoring"):
                checks = await self.health_service.perform_comprehensive_health_check()
                
                # Determine overall status
                statuses = [check.status for check in checks.values()]
                if HealthStatus.UNHEALTHY in statuses:
                    overall_status = HealthStatus.UNHEALTHY
                elif HealthStatus.DEGRADED in statuses:
                    overall_status = HealthStatus.DEGRADED
                else:
                    overall_status = HealthStatus.HEALTHY
                
                return {
                    "status": overall_status.value,
                    "timestamp": datetime.utcnow().isoformat(),
                    "checks": {name: {
                        "status": check.status.value,
                        "message": check.message,
                        "duration_ms": check.duration_ms,
                        "details": check.details
                    } for name, check in checks.items()}
                }
                
        except Exception as e:
            self.report_error(e, "monitoring", "health_check")
            return {
                "status": "error",
                "message": f"Health check failed: {e}",
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of monitoring metrics.
        
        Returns:
            Metrics summary
        """
        now = datetime.utcnow()
        hour_ago = now - timedelta(hours=1)
        
        # Recent logs
        recent_logs = [log for log in self.log_entries if log.timestamp >= hour_ago]
        log_levels = {}
        for level in LogLevel:
            log_levels[level.value] = len([log for log in recent_logs if log.level == level])
        
        # Recent errors
        recent_errors = [err for err in self.error_reports if err.timestamp >= hour_ago]
        error_types = {}
        for error in recent_errors:
            error_types[error.error_type] = error_types.get(error.error_type, 0) + 1
        
        # Recent audit events
        recent_audits = [audit for audit in self.audit_events if audit.timestamp >= hour_ago]
        
        return {
            "timestamp": now.isoformat(),
            "monitoring_period": "last_hour",
            "logs": {
                "total": len(recent_logs),
                "by_level": log_levels
            },
            "errors": {
                "total": len(recent_errors),
                "by_type": error_types,
                "unresolved": len([err for err in recent_errors if not err.resolved])
            },
            "audit_events": {
                "total": len(recent_audits)
            },
            "storage_usage": {
                "log_entries": f"{len(self.log_entries)}/{self.max_log_entries}",
                "error_reports": f"{len(self.error_reports)}/{self.max_error_reports}",
                "audit_events": f"{len(self.audit_events)}/{self.max_audit_events}"
            }
        }
    
    def export_logs(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        level: Optional[LogLevel] = None,
        component: Optional[str] = None,
        operation: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Export filtered logs.
        
        Args:
            start_time: Start time filter
            end_time: End time filter
            level: Log level filter
            component: Component filter
            operation: Operation filter
            
        Returns:
            Filtered log entries
        """
        filtered_logs = self.log_entries
        
        if start_time:
            filtered_logs = [log for log in filtered_logs if log.timestamp >= start_time]
        
        if end_time:
            filtered_logs = [log for log in filtered_logs if log.timestamp <= end_time]
        
        if level:
            filtered_logs = [log for log in filtered_logs if log.level == level]
        
        if component:
            filtered_logs = [log for log in filtered_logs if log.component == component]
        
        if operation:
            filtered_logs = [log for log in filtered_logs if log.operation == operation]
        
        return [log.to_dict() for log in filtered_logs]
    
    def resolve_error(self, error_id: str, resolution_notes: str) -> bool:
        """Mark an error as resolved.
        
        Args:
            error_id: Error ID to resolve
            resolution_notes: Notes about the resolution
            
        Returns:
            True if error was found and resolved
        """
        for error in self.error_reports:
            if error.error_id == error_id:
                error.resolved = True
                error.resolution_notes = resolution_notes
                
                self.info(
                    f"Error resolved: {error_id}",
                    operation="error_resolution",
                    component="monitoring",
                    error_id=error_id,
                    resolution_notes=resolution_notes
                )
                return True
        
        return False


# Global monitor instance
_monitor_instance: Optional[ProductionMonitor] = None


def get_monitor() -> ProductionMonitor:
    """Get global monitor instance."""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = ProductionMonitor()
    return _monitor_instance


def init_monitor(
    log_level: LogLevel = LogLevel.INFO,
    log_file: Optional[Path] = None,
    **kwargs
) -> ProductionMonitor:
    """Initialize global monitor instance."""
    global _monitor_instance
    _monitor_instance = ProductionMonitor(
        log_level=log_level,
        log_file=log_file,
        **kwargs
    )
    return _monitor_instance


# Convenience functions
def log_info(message: str, **kwargs) -> None:
    """Log info message using global monitor."""
    get_monitor().info(message, **kwargs)


def log_error(message: str, **kwargs) -> None:
    """Log error message using global monitor."""
    get_monitor().error(message, **kwargs)


def log_warning(message: str, **kwargs) -> None:
    """Log warning message using global monitor."""
    get_monitor().warning(message, **kwargs)


def report_error(error: Exception, component: str, operation: str, **kwargs) -> str:
    """Report error using global monitor."""
    return get_monitor().report_error(error, component, operation, **kwargs)


def audit_event(event_type: str, operation: str, resource: str, action: str, **kwargs) -> str:
    """Record audit event using global monitor."""
    return get_monitor().audit_event(event_type, operation, resource, action, **kwargs)


def monitor_operation(operation: str, component: str, **kwargs):
    """Monitor operation using global monitor."""
    return get_monitor().monitor_operation(operation, component, **kwargs)


def monitor_async_operation(operation: str, component: str, **kwargs):
    """Monitor async operation using global monitor."""
    return get_monitor().monitor_async_operation(operation, component, **kwargs)