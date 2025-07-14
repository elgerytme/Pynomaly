#!/usr/bin/env python3
"""Error tracking integration for real-time monitoring dashboard."""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pynomaly.shared.logging import get_logger

logger = get_logger(__name__)


class ErrorEvent:
    """Represents an error event in the system."""
    
    def __init__(
        self,
        error_id: str,
        timestamp: datetime,
        error_type: str,
        message: str,
        component: str,
        severity: str = "error",
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        stack_trace: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.error_id = error_id
        self.timestamp = timestamp
        self.error_type = error_type
        self.message = message
        self.component = component
        self.severity = severity  # critical, error, warning, info
        self.user_id = user_id
        self.session_id = session_id
        self.stack_trace = stack_trace
        self.metadata = metadata or {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert error event to dictionary."""
        return {
            'error_id': self.error_id,
            'timestamp': self.timestamp.isoformat(),
            'error_type': self.error_type,
            'message': self.message,
            'component': self.component,
            'severity': self.severity,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'stack_trace': self.stack_trace,
            'metadata': self.metadata,
        }


class ErrorTracker:
    """Enhanced error tracking system with real-time capabilities."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.error_history: List[ErrorEvent] = []
        self.error_counts = {
            'critical': 0,
            'error': 0,
            'warning': 0,
            'info': 0,
        }
        self.component_errors: Dict[str, int] = {}
        self.error_rate_window = timedelta(minutes=5)
        self.subscribers: List[callable] = []
        
    def track_error(
        self,
        error: Exception,
        component: str,
        severity: str = "error",
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Track an error event."""
        error_id = str(uuid4())
        timestamp = datetime.utcnow()
        
        # Extract error information
        error_type = type(error).__name__
        message = str(error)
        
        # Create error event
        error_event = ErrorEvent(
            error_id=error_id,
            timestamp=timestamp,
            error_type=error_type,
            message=message,
            component=component,
            severity=severity,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata
        )
        
        # Store error
        self._store_error(error_event)
        
        # Notify subscribers
        asyncio.create_task(self._notify_subscribers(error_event))
        
        logger.error(f"Error tracked: {error_id} - {component} - {message}")
        return error_id
    
    def _store_error(self, error_event: ErrorEvent):
        """Store error event in history."""
        self.error_history.append(error_event)
        
        # Maintain max history size
        if len(self.error_history) > self.max_history:
            removed_error = self.error_history.pop(0)
            self.error_counts[removed_error.severity] -= 1
            self.component_errors[removed_error.component] -= 1
        
        # Update counters
        self.error_counts[error_event.severity] += 1
        if error_event.component in self.component_errors:
            self.component_errors[error_event.component] += 1
        else:
            self.component_errors[error_event.component] = 1
    
    async def _notify_subscribers(self, error_event: ErrorEvent):
        """Notify all subscribers of new error event."""
        for subscriber in self.subscribers:
            try:
                if asyncio.iscoroutinefunction(subscriber):
                    await subscriber(error_event)
                else:
                    subscriber(error_event)
            except Exception as e:
                logger.warning(f"Error notifying subscriber: {e}")
    
    def get_error_summary(self, time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """Get error summary statistics."""
        if time_window is None:
            time_window = self.error_rate_window
        
        cutoff_time = datetime.utcnow() - time_window
        recent_errors = [
            error for error in self.error_history
            if error.timestamp >= cutoff_time
        ]
        
        recent_counts = {'critical': 0, 'error': 0, 'warning': 0, 'info': 0}
        
        for error in recent_errors:
            recent_counts[error.severity] += 1
        
        # Calculate error rate (errors per minute)
        total_recent_errors = len(recent_errors)
        window_minutes = time_window.total_seconds() / 60
        error_rate = total_recent_errors / window_minutes if window_minutes > 0 else 0
        
        return {
            'total_errors': len(self.error_history),
            'recent_errors': total_recent_errors,
            'error_rate_per_minute': round(error_rate, 2),
            'error_counts': dict(self.error_counts),
            'recent_counts': recent_counts,
            'time_window_minutes': window_minutes,
            'last_updated': datetime.utcnow().isoformat(),
        }
    
    def get_recent_errors(self, limit: int = 50, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get recent error events."""
        errors = self.error_history
        
        if severity:
            errors = [e for e in errors if e.severity == severity]
        
        # Sort by timestamp descending and limit
        recent_errors = sorted(errors, key=lambda x: x.timestamp, reverse=True)[:limit]
        
        return [error.to_dict() for error in recent_errors]


class HealthMonitorIntegration:
    """Health monitoring integration for real-time dashboard."""
    
    def __init__(self, error_tracker: ErrorTracker):
        self.error_tracker = error_tracker
        self.health_checks = {}
        self.system_status = "healthy"
        self.last_health_check = None
        
    def register_health_check(self, name: str, check_func: callable, critical: bool = False):
        """Register a health check."""
        self.health_checks[name] = {
            'func': check_func,
            'critical': critical,
            'last_result': None,
            'last_check': None,
        }
    
    async def run_health_checks(self) -> Dict[str, Any]:
        """Run all registered health checks."""
        self.last_health_check = datetime.utcnow()
        results = {}
        overall_status = "healthy"
        
        for name, check_info in self.health_checks.items():
            try:
                if asyncio.iscoroutinefunction(check_info['func']):
                    result = await check_info['func']()
                else:
                    result = check_info['func']()
                
                check_info['last_result'] = result
                check_info['last_check'] = self.last_health_check
                results[name] = {
                    'status': 'ok' if result else 'failed',
                    'critical': check_info['critical'],
                    'last_check': self.last_health_check.isoformat(),
                }
                
                if not result and check_info['critical']:
                    overall_status = "critical"
                        
            except Exception as e:
                results[name] = {
                    'status': 'error',
                    'error': str(e),
                    'critical': check_info['critical'],
                    'last_check': self.last_health_check.isoformat(),
                }
                if check_info['critical']:
                    overall_status = "critical"
        
        self.system_status = overall_status
        
        return {
            'overall_status': overall_status,
            'checks': results,
            'last_updated': self.last_health_check.isoformat(),
        }
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get system health summary."""
        error_summary = self.error_tracker.get_error_summary()
        
        # Calculate health score based on error rates
        health_score = 100
        error_rate = error_summary.get('error_rate_per_minute', 0)
        if error_rate > 5:
            health_score -= 30
        elif error_rate > 1:
            health_score -= 15
        
        return {
            'health_score': max(0, health_score),
            'system_status': self.system_status,
            'error_summary': error_summary,
            'last_health_check': self.last_health_check.isoformat() if self.last_health_check else None,
            'generated_at': datetime.utcnow().isoformat(),
        }


# Global instances
_error_tracker = None
_health_monitor = None


def get_error_tracker() -> ErrorTracker:
    """Get the global error tracker instance."""
    global _error_tracker
    if _error_tracker is None:
        _error_tracker = ErrorTracker()
    return _error_tracker


def get_health_monitor() -> HealthMonitorIntegration:
    """Get the global health monitor instance."""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = HealthMonitorIntegration(get_error_tracker())
    return _health_monitor