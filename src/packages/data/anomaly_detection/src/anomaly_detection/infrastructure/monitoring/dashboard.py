"""Monitoring dashboard module for anomaly detection system."""

from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import structlog

logger = structlog.get_logger(__name__)


class MonitoringDashboard:
    """Simple monitoring dashboard for displaying system metrics."""
    
    def __init__(self):
        self.data = {}
        logger.info("MonitoringDashboard initialized")
    
    def update_data(self, key: str, value: Any) -> None:
        """Update dashboard data."""
        self.data[key] = value
    
    def get_data(self) -> Dict[str, Any]:
        """Get current dashboard data."""
        return self.data.copy()
    
    def clear_data(self) -> None:
        """Clear dashboard data."""
        self.data.clear()


# Global instance
_monitoring_dashboard: Optional[MonitoringDashboard] = None


def get_monitoring_dashboard() -> MonitoringDashboard:
    """Get the global monitoring dashboard instance."""
    global _monitoring_dashboard
    
    if _monitoring_dashboard is None:
        _monitoring_dashboard = MonitoringDashboard()
    
    return _monitoring_dashboard