"""Performance monitoring module for tracking system performance."""

import time
import psutil
from typing import Dict, Any, Optional, List, ContextManager
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from contextlib import contextmanager
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    timestamp: datetime
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    duration_ms: float
    operation_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceContext:
    """Context for tracking performance of operations."""
    
    def __init__(self, operation_name: str, monitor: 'PerformanceMonitor'):
        self.operation_name = operation_name
        self.monitor = monitor
        self.start_time: Optional[float] = None
        self.start_cpu: Optional[float] = None
        self.start_memory: Optional[float] = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.start_cpu = psutil.cpu_percent()
        self.start_memory = psutil.virtual_memory().percent
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration_ms = (time.time() - self.start_time) * 1000
            end_cpu = psutil.cpu_percent()
            end_memory = psutil.virtual_memory()
            
            metrics = PerformanceMetrics(
                timestamp=datetime.utcnow(),
                cpu_percent=end_cpu,
                memory_mb=end_memory.used / 1024 / 1024,
                memory_percent=end_memory.percent,
                duration_ms=duration_ms,
                operation_name=self.operation_name
            )
            
            self.monitor.record_metrics(metrics)


class PerformanceMonitor:
    """System performance monitor."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics_history: List[PerformanceMetrics] = []
        logger.info("PerformanceMonitor initialized", max_history=max_history)
    
    def record_metrics(self, metrics: PerformanceMetrics) -> None:
        """Record performance metrics."""
        self.metrics_history.append(metrics)
        
        # Keep history bounded
        if len(self.metrics_history) > self.max_history:
            self.metrics_history = self.metrics_history[-self.max_history:]
        
        logger.debug("Performance metrics recorded",
                    operation=metrics.operation_name,
                    duration_ms=metrics.duration_ms,
                    cpu_percent=metrics.cpu_percent,
                    memory_mb=metrics.memory_mb)
    
    @contextmanager
    def track_operation(self, operation_name: str) -> PerformanceContext:
        """Context manager for tracking operation performance."""
        context = PerformanceContext(operation_name, self)
        with context:
            yield context
    
    def get_metrics_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get performance metrics summary for the specified time period."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        recent_metrics = [
            m for m in self.metrics_history
            if m.timestamp >= cutoff_time
        ]
        
        if not recent_metrics:
            return {"message": "No metrics available for the specified period"}
        
        durations = [m.duration_ms for m in recent_metrics]
        cpu_values = [m.cpu_percent for m in recent_metrics]
        memory_values = [m.memory_mb for m in recent_metrics]
        
        return {
            "period_hours": hours,
            "total_operations": len(recent_metrics),
            "average_duration_ms": sum(durations) / len(durations),
            "max_duration_ms": max(durations),
            "min_duration_ms": min(durations),
            "average_cpu_percent": sum(cpu_values) / len(cpu_values),
            "average_memory_mb": sum(memory_values) / len(memory_values),
            "operations_per_hour": len(recent_metrics) / hours
        }
    
    def get_operation_stats(self, operation_name: str) -> Dict[str, Any]:
        """Get statistics for a specific operation."""
        operation_metrics = [
            m for m in self.metrics_history
            if m.operation_name == operation_name
        ]
        
        if not operation_metrics:
            return {"message": f"No metrics found for operation '{operation_name}'"}
        
        durations = [m.duration_ms for m in operation_metrics]
        
        return {
            "operation_name": operation_name,
            "total_executions": len(operation_metrics),
            "average_duration_ms": sum(durations) / len(durations),
            "max_duration_ms": max(durations),
            "min_duration_ms": min(durations),
            "last_execution": operation_metrics[-1].timestamp.isoformat()
        }
    
    def clear_history(self) -> None:
        """Clear performance metrics history."""
        self.metrics_history.clear()
        logger.info("Performance metrics history cleared")


# Global instance
_performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    global _performance_monitor
    
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    
    return _performance_monitor