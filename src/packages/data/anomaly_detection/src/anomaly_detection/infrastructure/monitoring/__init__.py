"""Monitoring and observability components."""

from ..observability.metrics import MetricsCollector, get_metrics_collector

# Import migrated monitoring functions from shared.observability 
try:
    from shared.observability.infrastructure.monitoring.health_checker import get_health_checker
    from shared.observability.infrastructure.monitoring.performance_monitor import get_performance_monitor
    from shared.observability.infrastructure.dashboards.monitoring_dashboard import get_monitoring_dashboard
except ImportError:
    # Fallback implementations for missing functions
    def get_health_checker():
        """Fallback health checker implementation."""
        from ..observability.metrics import get_metrics_collector
        return get_metrics_collector()
    
    def get_performance_monitor():
        """Fallback performance monitor implementation.""" 
        from ..observability.metrics import get_metrics_collector
        return get_metrics_collector()
        
    def get_monitoring_dashboard():
        """Fallback monitoring dashboard implementation."""
        from ..observability.metrics import get_metrics_collector
        return get_metrics_collector()

__all__ = [
    "MetricsCollector",
    "get_metrics_collector", 
    "get_health_checker",
    "get_performance_monitor",
    "get_monitoring_dashboard",
]