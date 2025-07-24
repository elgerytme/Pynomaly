"""Alerting system for monitoring anomaly detection operations."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(str, Enum):
    """Alert status."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


@dataclass
class Alert:
    """Alert data structure."""
    
    alert_id: str
    title: str
    message: str
    severity: AlertSeverity
    status: AlertStatus = AlertStatus.ACTIVE
    source: str = "anomaly_detection"
    created_at: datetime = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}


class AlertingSystem:
    """System for managing alerts and notifications."""
    
    def __init__(self):
        """Initialize alerting system."""
        self.alerts: Dict[str, Alert] = {}
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        
    def create_alert(
        self,
        title: str,
        message: str,
        severity: AlertSeverity,
        source: str = "anomaly_detection",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new alert."""
        alert_id = f"alert_{int(datetime.utcnow().timestamp())}"
        
        alert = Alert(
            alert_id=alert_id,
            title=title,
            message=message,
            severity=severity,
            source=source,
            metadata=metadata or {}
        )
        
        self.alerts[alert_id] = alert
        logger.info(f"Alert created: {alert_id} - {title}")
        
        return alert_id
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        if alert_id in self.alerts:
            self.alerts[alert_id].status = AlertStatus.ACKNOWLEDGED
            self.alerts[alert_id].acknowledged_at = datetime.utcnow()
            logger.info(f"Alert acknowledged: {alert_id}")
            return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        if alert_id in self.alerts:
            self.alerts[alert_id].status = AlertStatus.RESOLVED
            self.alerts[alert_id].resolved_at = datetime.utcnow()
            logger.info(f"Alert resolved: {alert_id}")
            return True
        return False
    
    def get_alert(self, alert_id: str) -> Optional[Alert]:
        """Get an alert by ID."""
        return self.alerts.get(alert_id)
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return [
            alert for alert in self.alerts.values()
            if alert.status == AlertStatus.ACTIVE
        ]
    
    def get_alerts_by_severity(self, severity: AlertSeverity) -> List[Alert]:
        """Get alerts by severity level."""
        return [
            alert for alert in self.alerts.values()
            if alert.severity == severity
        ]
    
    def clear_resolved_alerts(self) -> int:
        """Clear resolved alerts and return count."""
        resolved_alerts = [
            alert_id for alert_id, alert in self.alerts.items()
            if alert.status == AlertStatus.RESOLVED
        ]
        
        for alert_id in resolved_alerts:
            del self.alerts[alert_id]
        
        logger.info(f"Cleared {len(resolved_alerts)} resolved alerts")
        return len(resolved_alerts)
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of all alerts."""
        summary = {
            "total_alerts": len(self.alerts),
            "active_alerts": len(self.get_active_alerts()),
            "by_severity": {
                "low": len(self.get_alerts_by_severity(AlertSeverity.LOW)),
                "medium": len(self.get_alerts_by_severity(AlertSeverity.MEDIUM)),
                "high": len(self.get_alerts_by_severity(AlertSeverity.HIGH)),
                "critical": len(self.get_alerts_by_severity(AlertSeverity.CRITICAL))
            },
            "by_status": {
                "active": len([a for a in self.alerts.values() if a.status == AlertStatus.ACTIVE]),
                "acknowledged": len([a for a in self.alerts.values() if a.status == AlertStatus.ACKNOWLEDGED]),
                "resolved": len([a for a in self.alerts.values() if a.status == AlertStatus.RESOLVED])
            }
        }
        return summary


# Global alerting system instance
_alerting_system = None

def get_alerting_system() -> AlertingSystem:
    """Get or create the global alerting system instance."""
    global _alerting_system
    if _alerting_system is None:
        _alerting_system = AlertingSystem()
    return _alerting_system