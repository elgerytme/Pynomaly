"""
Security monitoring dashboard for Web UI
Provides real-time security metrics, alerts, and threat analysis
"""

import asyncio
import json
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

try:
    from monorepo.presentation.web.enhanced_auth import (
        AuthenticationMethod,
        Permission,
        get_auth_service,
    )
    from monorepo.presentation.web.security_features import (
        SecurityEvent,
        SecurityEventType,
        SecurityThreatLevel,
        get_security_middleware,
    )
except ImportError:
    # Fallback for testing
    from enum import Enum

    class Permission(Enum):
        SYSTEM_MONITOR = "system:monitor"

    class SecurityEventType(Enum):
        RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"

    class SecurityThreatLevel(Enum):
        LOW = "low"
        HIGH = "high"


class AlertSeverity(Enum):
    """Alert severity levels"""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertType(Enum):
    """Types of security alerts"""

    BRUTE_FORCE_ATTACK = "brute_force_attack"
    DDoS_ATTACK = "ddos_attack"
    SUSPICIOUS_IP = "suspicious_ip"
    MULTIPLE_FAILED_LOGINS = "multiple_failed_logins"
    UNUSUAL_TRAFFIC_PATTERN = "unusual_traffic_pattern"
    SECURITY_POLICY_VIOLATION = "security_policy_violation"
    POTENTIAL_DATA_BREACH = "potential_data_breach"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    MALICIOUS_REQUEST = "malicious_request"


@dataclass
class SecurityAlert:
    """Security alert data structure"""

    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    description: str
    affected_resources: list[str]
    source_ip: str
    timestamp: datetime
    is_acknowledged: bool = False
    acknowledged_by: str | None = None
    acknowledged_at: datetime | None = None
    auto_resolved: bool = False
    resolution_notes: str | None = None


@dataclass
class SecurityMetrics:
    """Security metrics snapshot"""

    timestamp: datetime
    total_requests: int
    blocked_requests: int
    failed_authentications: int
    active_sessions: int
    unique_ips: int
    suspicious_ips: int
    security_events: dict[str, int]
    threat_levels: dict[str, int]
    authentication_methods: dict[str, int]
    geographic_distribution: dict[str, int]


@dataclass
class ThreatIntelligence:
    """Threat intelligence data"""

    ip_address: str
    threat_score: int
    threat_categories: list[str]
    country: str
    organization: str
    last_seen: datetime
    reputation_sources: list[str]
    is_malicious: bool
    confidence_score: float


class SecurityMonitoringService:
    """Enhanced security monitoring service"""

    def __init__(self):
        self.alerts: dict[str, SecurityAlert] = {}
        self.metrics_history: deque[SecurityMetrics] = deque(
            maxlen=1440
        )  # 24 hours of minutes
        self.websocket_connections: list[WebSocket] = []
        self.threat_intelligence: dict[str, ThreatIntelligence] = {}

        # Alert thresholds
        self.alert_thresholds = {
            "failed_logins_per_minute": 10,
            "failed_logins_per_ip": 5,
            "requests_per_minute": 1000,
            "blocked_requests_ratio": 0.1,
            "unique_ips_per_minute": 100,
            "suspicious_patterns": 5,
        }

        # Traffic analysis
        self.traffic_analyzer = TrafficAnalyzer()

        # Start background tasks
        self.start_monitoring_tasks()

    def start_monitoring_tasks(self):
        """Start background monitoring tasks"""
        try:
            asyncio.create_task(self._collect_metrics_task())
            asyncio.create_task(self._analyze_threats_task())
            asyncio.create_task(self._cleanup_old_data_task())
        except RuntimeError:
            # No event loop running
            pass

    async def _collect_metrics_task(self):
        """Background task to collect security metrics"""
        while True:
            try:
                metrics = self.collect_current_metrics()
                self.metrics_history.append(metrics)

                # Analyze for alerts
                await self.analyze_for_alerts(metrics)

                # Broadcast to WebSocket clients
                await self.broadcast_metrics(metrics)

                await asyncio.sleep(60)  # Collect every minute

            except Exception as e:
                print(f"Error in metrics collection: {e}")
                await asyncio.sleep(60)

    async def _analyze_threats_task(self):
        """Background task to analyze threats"""
        while True:
            try:
                await self.analyze_threat_patterns()
                await asyncio.sleep(300)  # Analyze every 5 minutes

            except Exception as e:
                print(f"Error in threat analysis: {e}")
                await asyncio.sleep(300)

    async def _cleanup_old_data_task(self):
        """Background task to clean up old data"""
        while True:
            try:
                await self.cleanup_old_data()
                await asyncio.sleep(3600)  # Cleanup every hour

            except Exception as e:
                print(f"Error in data cleanup: {e}")
                await asyncio.sleep(3600)

    def collect_current_metrics(self) -> SecurityMetrics:
        """Collect current security metrics"""
        try:
            security_middleware = get_security_middleware()
            auth_service = get_auth_service()

            now = datetime.utcnow()

            # Get security events from the last minute
            recent_events = [
                event
                for event in security_middleware.security_events
                if (now - event["timestamp"]).total_seconds() < 60
            ]

            # Count events by type
            event_counts = defaultdict(int)
            threat_counts = defaultdict(int)

            for event in recent_events:
                event_counts[event["event_type"]] += 1
                threat_counts[event["threat_level"]] += 1

            # Get authentication metrics
            auth_metrics = auth_service.get_security_metrics()

            # Analyze IP patterns
            unique_ips = set()
            suspicious_ips = set()

            for event in recent_events:
                ip = event.get("ip_address")
                if ip:
                    unique_ips.add(ip)
                    if event.get("blocked", False):
                        suspicious_ips.add(ip)

            return SecurityMetrics(
                timestamp=now,
                total_requests=len(recent_events),
                blocked_requests=sum(
                    1 for e in recent_events if e.get("blocked", False)
                ),
                failed_authentications=auth_metrics.get("failed_login_attempts", 0),
                active_sessions=auth_metrics.get("active_sessions", 0),
                unique_ips=len(unique_ips),
                suspicious_ips=len(suspicious_ips),
                security_events=dict(event_counts),
                threat_levels=dict(threat_counts),
                authentication_methods=auth_metrics.get("session_methods", {}),
                geographic_distribution={},  # Would need GeoIP integration
            )

        except Exception as e:
            print(f"Error collecting metrics: {e}")
            return SecurityMetrics(
                timestamp=datetime.utcnow(),
                total_requests=0,
                blocked_requests=0,
                failed_authentications=0,
                active_sessions=0,
                unique_ips=0,
                suspicious_ips=0,
                security_events={},
                threat_levels={},
                authentication_methods={},
                geographic_distribution={},
            )

    async def analyze_for_alerts(self, metrics: SecurityMetrics):
        """Analyze metrics for potential security alerts"""
        alerts_to_create = []

        # Check for brute force attacks
        if (
            metrics.failed_authentications
            > self.alert_thresholds["failed_logins_per_minute"]
        ):
            alerts_to_create.append(
                {
                    "type": AlertType.BRUTE_FORCE_ATTACK,
                    "severity": AlertSeverity.CRITICAL,
                    "title": "Potential Brute Force Attack Detected",
                    "description": f"High number of failed authentication attempts: {metrics.failed_authentications}",
                    "affected_resources": ["authentication_system"],
                }
            )

        # Check for DDoS patterns
        if metrics.total_requests > self.alert_thresholds["requests_per_minute"]:
            alerts_to_create.append(
                {
                    "type": AlertType.DDoS_ATTACK,
                    "severity": AlertSeverity.WARNING,
                    "title": "High Traffic Volume Detected",
                    "description": f"Unusually high request volume: {metrics.total_requests} requests/minute",
                    "affected_resources": ["web_application"],
                }
            )

        # Check blocked request ratio
        if metrics.total_requests > 0:
            blocked_ratio = metrics.blocked_requests / metrics.total_requests
            if blocked_ratio > self.alert_thresholds["blocked_requests_ratio"]:
                alerts_to_create.append(
                    {
                        "type": AlertType.SECURITY_POLICY_VIOLATION,
                        "severity": AlertSeverity.WARNING,
                        "title": "High Security Violation Rate",
                        "description": f"High percentage of blocked requests: {blocked_ratio:.1%}",
                        "affected_resources": ["security_filters"],
                    }
                )

        # Check for suspicious IP patterns
        if metrics.suspicious_ips > 5:
            alerts_to_create.append(
                {
                    "type": AlertType.SUSPICIOUS_IP,
                    "severity": AlertSeverity.WARNING,
                    "title": "Multiple Suspicious IPs Detected",
                    "description": f"Multiple suspicious IP addresses detected: {metrics.suspicious_ips}",
                    "affected_resources": ["network_security"],
                }
            )

        # Create alerts
        for alert_data in alerts_to_create:
            await self.create_alert(**alert_data)

    async def create_alert(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        title: str,
        description: str,
        affected_resources: list[str],
        source_ip: str = "unknown",
    ) -> SecurityAlert:
        """Create new security alert"""
        from uuid import uuid4

        alert = SecurityAlert(
            alert_id=str(uuid4()),
            alert_type=alert_type,
            severity=severity,
            title=title,
            description=description,
            affected_resources=affected_resources,
            source_ip=source_ip,
            timestamp=datetime.utcnow(),
        )

        self.alerts[alert.alert_id] = alert

        # Broadcast alert to WebSocket clients
        await self.broadcast_alert(alert)

        return alert

    async def acknowledge_alert(self, alert_id: str, user_id: str) -> bool:
        """Acknowledge security alert"""
        if alert_id in self.alerts:
            alert = self.alerts[alert_id]
            alert.is_acknowledged = True
            alert.acknowledged_by = user_id
            alert.acknowledged_at = datetime.utcnow()

            # Broadcast update
            await self.broadcast_alert_update(alert)
            return True

        return False

    async def analyze_threat_patterns(self):
        """Analyze patterns to identify potential threats"""
        # Analyze recent events for patterns
        try:
            security_middleware = get_security_middleware()
            recent_events = list(security_middleware.security_events)[
                -1000:
            ]  # Last 1000 events

            # Group events by IP
            ip_events = defaultdict(list)
            for event in recent_events:
                ip = event.get("ip_address")
                if ip:
                    ip_events[ip].append(event)

            # Analyze each IP for suspicious patterns
            for ip, events in ip_events.items():
                await self._analyze_ip_threat_patterns(ip, events)

        except Exception as e:
            print(f"Error in threat pattern analysis: {e}")

    async def _analyze_ip_threat_patterns(self, ip: str, events: list[dict]):
        """Analyze threat patterns for specific IP"""
        if len(events) < 5:  # Not enough data
            return

        # Calculate threat score
        threat_score = 0
        threat_categories = []

        # Check for rapid requests
        time_spans = []
        for i in range(1, len(events)):
            if i < len(events):
                time_diff = events[i]["timestamp"] - events[i - 1]["timestamp"]
                time_spans.append(time_diff.total_seconds())

        if (
            time_spans and sum(time_spans) / len(time_spans) < 1
        ):  # Less than 1 second between requests
            threat_score += 30
            threat_categories.append("rapid_requests")

        # Check for blocked requests
        blocked_count = sum(1 for event in events if event.get("blocked", False))
        if blocked_count > len(events) * 0.5:  # More than 50% blocked
            threat_score += 40
            threat_categories.append("high_block_rate")

        # Check for security violations
        security_events = [
            event
            for event in events
            if event.get("event_type")
            in ["sql_injection_attempt", "xss_attempt", "suspicious_pattern"]
        ]
        if security_events:
            threat_score += len(security_events) * 10
            threat_categories.append("security_violations")

        # Update threat intelligence
        if threat_score > 50:
            self.threat_intelligence[ip] = ThreatIntelligence(
                ip_address=ip,
                threat_score=threat_score,
                threat_categories=threat_categories,
                country="unknown",  # Would need GeoIP integration
                organization="unknown",
                last_seen=datetime.utcnow(),
                reputation_sources=["internal_analysis"],
                is_malicious=threat_score > 80,
                confidence_score=min(threat_score / 100, 1.0),
            )

    async def cleanup_old_data(self):
        """Clean up old alerts and data"""
        cutoff_time = datetime.utcnow() - timedelta(days=7)

        # Remove old acknowledged alerts
        old_alerts = [
            alert_id
            for alert_id, alert in self.alerts.items()
            if alert.is_acknowledged
            and alert.acknowledged_at
            and alert.acknowledged_at < cutoff_time
        ]

        for alert_id in old_alerts:
            del self.alerts[alert_id]

        # Clean up old threat intelligence
        old_threats = [
            ip
            for ip, threat in self.threat_intelligence.items()
            if threat.last_seen < cutoff_time
        ]

        for ip in old_threats:
            del self.threat_intelligence[ip]

    async def connect_websocket(self, websocket: WebSocket):
        """Connect WebSocket client"""
        await websocket.accept()
        self.websocket_connections.append(websocket)

        # Send current metrics
        if self.metrics_history:
            latest_metrics = self.metrics_history[-1]
            await websocket.send_text(
                json.dumps(
                    {"type": "metrics", "data": asdict(latest_metrics)}, default=str
                )
            )

    def disconnect_websocket(self, websocket: WebSocket):
        """Disconnect WebSocket client"""
        if websocket in self.websocket_connections:
            self.websocket_connections.remove(websocket)

    async def broadcast_metrics(self, metrics: SecurityMetrics):
        """Broadcast metrics to all connected WebSocket clients"""
        if not self.websocket_connections:
            return

        message = json.dumps({"type": "metrics", "data": asdict(metrics)}, default=str)

        disconnected = []
        for websocket in self.websocket_connections:
            try:
                await websocket.send_text(message)
            except:
                disconnected.append(websocket)

        # Remove disconnected clients
        for websocket in disconnected:
            self.disconnect_websocket(websocket)

    async def broadcast_alert(self, alert: SecurityAlert):
        """Broadcast new alert to WebSocket clients"""
        if not self.websocket_connections:
            return

        message = json.dumps({"type": "alert", "data": asdict(alert)}, default=str)

        disconnected = []
        for websocket in self.websocket_connections:
            try:
                await websocket.send_text(message)
            except:
                disconnected.append(websocket)

        # Remove disconnected clients
        for websocket in disconnected:
            self.disconnect_websocket(websocket)

    async def broadcast_alert_update(self, alert: SecurityAlert):
        """Broadcast alert update to WebSocket clients"""
        if not self.websocket_connections:
            return

        message = json.dumps(
            {"type": "alert_update", "data": asdict(alert)}, default=str
        )

        disconnected = []
        for websocket in self.websocket_connections:
            try:
                await websocket.send_text(message)
            except:
                disconnected.append(websocket)

        # Remove disconnected clients
        for websocket in disconnected:
            self.disconnect_websocket(websocket)

    def get_dashboard_data(self) -> dict[str, Any]:
        """Get comprehensive dashboard data"""
        # Get recent metrics (last hour)
        recent_metrics = (
            list(self.metrics_history)[-60:] if self.metrics_history else []
        )

        # Get active alerts
        active_alerts = [
            alert for alert in self.alerts.values() if not alert.is_acknowledged
        ]

        # Get top threats
        top_threats = sorted(
            self.threat_intelligence.values(),
            key=lambda x: x.threat_score,
            reverse=True,
        )[:10]

        # Calculate summary statistics
        if recent_metrics:
            total_requests = sum(m.total_requests for m in recent_metrics)
            total_blocked = sum(m.blocked_requests for m in recent_metrics)
            avg_active_sessions = sum(m.active_sessions for m in recent_metrics) / len(
                recent_metrics
            )
        else:
            total_requests = 0
            total_blocked = 0
            avg_active_sessions = 0

        return {
            "summary": {
                "total_requests_hour": total_requests,
                "blocked_requests_hour": total_blocked,
                "block_rate": (total_blocked / total_requests * 100)
                if total_requests > 0
                else 0,
                "active_alerts": len(active_alerts),
                "active_sessions": int(avg_active_sessions),
                "top_threats": len(top_threats),
            },
            "metrics_timeline": [asdict(m) for m in recent_metrics],
            "active_alerts": [asdict(alert) for alert in active_alerts],
            "top_threats": [asdict(threat) for threat in top_threats],
            "alert_distribution": {
                alert_type.value: sum(
                    1 for a in self.alerts.values() if a.alert_type == alert_type
                )
                for alert_type in AlertType
            },
            "threat_categories": {
                category: sum(
                    1
                    for t in self.threat_intelligence.values()
                    if category in t.threat_categories
                )
                for category in [
                    "rapid_requests",
                    "high_block_rate",
                    "security_violations",
                    "suspicious_patterns",
                ]
            },
        }


class TrafficAnalyzer:
    """Traffic pattern analyzer for anomaly detection"""

    def __init__(self):
        self.request_patterns = defaultdict(list)
        self.baseline_metrics = {}

    def analyze_request_pattern(self, ip: str, request_data: dict) -> dict[str, Any]:
        """Analyze request pattern for anomalies"""
        pattern = {
            "timestamp": datetime.utcnow(),
            "method": request_data.get("method", "GET"),
            "path": request_data.get("path", "/"),
            "user_agent": request_data.get("user_agent", ""),
            "size": request_data.get("size", 0),
        }

        self.request_patterns[ip].append(pattern)

        # Keep only recent patterns (last hour)
        cutoff = datetime.utcnow() - timedelta(hours=1)
        self.request_patterns[ip] = [
            p for p in self.request_patterns[ip] if p["timestamp"] > cutoff
        ]

        # Analyze patterns
        return self._analyze_ip_pattern(ip)

    def _analyze_ip_pattern(self, ip: str) -> dict[str, Any]:
        """Analyze patterns for specific IP"""
        patterns = self.request_patterns[ip]
        if len(patterns) < 10:
            return {"anomaly_score": 0, "patterns": []}

        anomaly_score = 0
        detected_patterns = []

        # Check request frequency
        time_diffs = []
        for i in range(1, len(patterns)):
            diff = (
                patterns[i]["timestamp"] - patterns[i - 1]["timestamp"]
            ).total_seconds()
            time_diffs.append(diff)

        if time_diffs:
            avg_interval = sum(time_diffs) / len(time_diffs)
            if avg_interval < 0.5:  # Less than 0.5 seconds between requests
                anomaly_score += 30
                detected_patterns.append("rapid_requests")

        # Check for path scanning
        unique_paths = set(p["path"] for p in patterns)
        if len(unique_paths) > len(patterns) * 0.8:  # High path diversity
            anomaly_score += 20
            detected_patterns.append("path_scanning")

        # Check for user agent consistency
        user_agents = set(p["user_agent"] for p in patterns)
        if len(user_agents) > 3:  # Multiple user agents
            anomaly_score += 15
            detected_patterns.append("user_agent_spoofing")

        return {
            "anomaly_score": anomaly_score,
            "patterns": detected_patterns,
            "request_count": len(patterns),
            "unique_paths": len(unique_paths),
            "avg_interval": sum(time_diffs) / len(time_diffs) if time_diffs else 0,
        }


# Global monitoring service instance
_monitoring_service: SecurityMonitoringService | None = None


def get_monitoring_service() -> SecurityMonitoringService:
    """Get global security monitoring service instance"""
    global _monitoring_service
    if _monitoring_service is None:
        _monitoring_service = SecurityMonitoringService()
    return _monitoring_service


# Create API router for security monitoring endpoints
def create_security_monitoring_router() -> APIRouter:
    """Create security monitoring API router"""
    router = APIRouter(prefix="/api/security", tags=["security-monitoring"])

    @router.get("/dashboard")
    async def get_security_dashboard(request: Request):
        """Get security dashboard data"""
        # Check permissions
        try:
            auth_service = get_auth_service()
            session = await auth_service.get_current_session(request)
            if not session or not auth_service.has_permission(
                session, Permission.SYSTEM_MONITOR
            ):
                from fastapi import HTTPException, status

                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="System monitoring permission required",
                )
        except ImportError:
            pass  # Skip permission check in testing

        monitoring_service = get_monitoring_service()
        return monitoring_service.get_dashboard_data()

    @router.get("/alerts")
    async def get_security_alerts(request: Request):
        """Get security alerts"""
        monitoring_service = get_monitoring_service()
        return {
            "alerts": [asdict(alert) for alert in monitoring_service.alerts.values()],
            "total": len(monitoring_service.alerts),
        }

    @router.post("/alerts/{alert_id}/acknowledge")
    async def acknowledge_alert(alert_id: str, request: Request):
        """Acknowledge security alert"""
        # Get user from session
        user_id = "system"  # Would get from session in production

        monitoring_service = get_monitoring_service()
        success = await monitoring_service.acknowledge_alert(alert_id, user_id)

        return {"success": success}

    @router.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for real-time security monitoring"""
        monitoring_service = get_monitoring_service()
        await monitoring_service.connect_websocket(websocket)

        try:
            while True:
                # Keep connection alive
                await websocket.receive_text()
        except WebSocketDisconnect:
            monitoring_service.disconnect_websocket(websocket)

    @router.get("/dashboard/html")
    async def get_security_dashboard_html():
        """Get security monitoring dashboard HTML"""
        html_content = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Security Monitoring Dashboard - Pynomaly</title>
            <script src="https://cdn.tailwindcss.com"></script>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                .alert-critical { @apply border-l-4 border-red-500 bg-red-50; }
                .alert-warning { @apply border-l-4 border-yellow-500 bg-yellow-50; }
                .alert-info { @apply border-l-4 border-blue-500 bg-blue-50; }
                .metric-card { @apply bg-white rounded-lg shadow p-6 transition-all hover:shadow-lg; }
                .threat-indicator { @apply inline-block w-3 h-3 rounded-full; }
                .threat-high { @apply bg-red-500; }
                .threat-medium { @apply bg-yellow-500; }
                .threat-low { @apply bg-green-500; }
            </style>
        </head>
        <body class="bg-gray-100 min-h-screen">
            <div class="container mx-auto px-4 py-8">
                <h1 class="text-3xl font-bold text-gray-800 mb-8">Security Monitoring Dashboard</h1>

                <!-- Summary Cards -->
                <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
                    <div class="metric-card">
                        <h3 class="text-sm font-medium text-gray-500">Requests (1h)</h3>
                        <p class="text-2xl font-bold text-gray-900" id="total-requests">-</p>
                    </div>
                    <div class="metric-card">
                        <h3 class="text-sm font-medium text-gray-500">Blocked Requests</h3>
                        <p class="text-2xl font-bold text-red-600" id="blocked-requests">-</p>
                    </div>
                    <div class="metric-card">
                        <h3 class="text-sm font-medium text-gray-500">Active Alerts</h3>
                        <p class="text-2xl font-bold text-yellow-600" id="active-alerts">-</p>
                    </div>
                    <div class="metric-card">
                        <h3 class="text-sm font-medium text-gray-500">Active Sessions</h3>
                        <p class="text-2xl font-bold text-green-600" id="active-sessions">-</p>
                    </div>
                </div>

                <!-- Charts -->
                <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
                    <div class="bg-white rounded-lg shadow p-6">
                        <h3 class="text-lg font-semibold mb-4">Request Timeline</h3>
                        <canvas id="requests-chart" width="400" height="200"></canvas>
                    </div>
                    <div class="bg-white rounded-lg shadow p-6">
                        <h3 class="text-lg font-semibold mb-4">Threat Distribution</h3>
                        <canvas id="threats-chart" width="400" height="200"></canvas>
                    </div>
                </div>

                <!-- Alerts -->
                <div class="bg-white rounded-lg shadow p-6 mb-8">
                    <h3 class="text-lg font-semibold mb-4">Active Security Alerts</h3>
                    <div id="alerts-container">
                        <p class="text-gray-500">No active alerts</p>
                    </div>
                </div>

                <!-- Top Threats -->
                <div class="bg-white rounded-lg shadow p-6">
                    <h3 class="text-lg font-semibold mb-4">Top Threats</h3>
                    <div id="threats-container">
                        <p class="text-gray-500">No threats detected</p>
                    </div>
                </div>
            </div>

            <script>
                // WebSocket connection for real-time updates
                const ws = new WebSocket(`ws://${window.location.host}/api/security/ws`);

                let requestsChart, threatsChart;

                // Initialize charts
                function initCharts() {
                    const requestsCtx = document.getElementById('requests-chart').getContext('2d');
                    requestsChart = new Chart(requestsCtx, {
                        type: 'line',
                        data: {
                            labels: [],
                            datasets: [{
                                label: 'Requests',
                                data: [],
                                borderColor: 'rgb(59, 130, 246)',
                                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                                tension: 0.1
                            }, {
                                label: 'Blocked',
                                data: [],
                                borderColor: 'rgb(239, 68, 68)',
                                backgroundColor: 'rgba(239, 68, 68, 0.1)',
                                tension: 0.1
                            }]
                        },
                        options: {
                            responsive: true,
                            scales: {
                                y: {
                                    beginAtZero: true
                                }
                            }
                        }
                    });

                    const threatsCtx = document.getElementById('threats-chart').getContext('2d');
                    threatsChart = new Chart(threatsCtx, {
                        type: 'doughnut',
                        data: {
                            labels: [],
                            datasets: [{
                                data: [],
                                backgroundColor: [
                                    'rgb(239, 68, 68)',
                                    'rgb(245, 158, 11)',
                                    'rgb(59, 130, 246)',
                                    'rgb(16, 185, 129)'
                                ]
                            }]
                        },
                        options: {
                            responsive: true
                        }
                    });
                }

                // Update dashboard with new data
                function updateDashboard(data) {
                    if (data.type === 'metrics') {
                        updateMetrics(data.data);
                    } else if (data.type === 'alert') {
                        addAlert(data.data);
                    }
                }

                function updateMetrics(metrics) {
                    // Update summary cards
                    document.getElementById('total-requests').textContent = metrics.total_requests;
                    document.getElementById('blocked-requests').textContent = metrics.blocked_requests;
                    document.getElementById('active-sessions').textContent = metrics.active_sessions;

                    // Update charts
                    if (requestsChart) {
                        const time = new Date(metrics.timestamp).toLocaleTimeString();

                        requestsChart.data.labels.push(time);
                        requestsChart.data.datasets[0].data.push(metrics.total_requests);
                        requestsChart.data.datasets[1].data.push(metrics.blocked_requests);

                        // Keep only last 20 data points
                        if (requestsChart.data.labels.length > 20) {
                            requestsChart.data.labels.shift();
                            requestsChart.data.datasets[0].data.shift();
                            requestsChart.data.datasets[1].data.shift();
                        }

                        requestsChart.update();
                    }
                }

                function addAlert(alert) {
                    const alertsContainer = document.getElementById('alerts-container');

                    const alertElement = document.createElement('div');
                    alertElement.className = `alert-${alert.severity} p-4 mb-4 rounded`;
                    alertElement.innerHTML = `
                        <div class="flex justify-between items-start">
                            <div>
                                <h4 class="font-semibold">${alert.title}</h4>
                                <p class="text-sm text-gray-600">${alert.description}</p>
                                <p class="text-xs text-gray-500 mt-1">
                                    ${new Date(alert.timestamp).toLocaleString()} - IP: ${alert.source_ip}
                                </p>
                            </div>
                            <button onclick="acknowledgeAlert('${alert.alert_id}')"
                                    class="bg-blue-500 text-white px-3 py-1 rounded text-sm hover:bg-blue-600">
                                Acknowledge
                            </button>
                        </div>
                    `;

                    alertsContainer.insertBefore(alertElement, alertsContainer.firstChild);

                    // Update alert count
                    const currentCount = parseInt(document.getElementById('active-alerts').textContent) || 0;
                    document.getElementById('active-alerts').textContent = currentCount + 1;
                }

                function acknowledgeAlert(alertId) {
                    fetch(`/api/security/alerts/${alertId}/acknowledge`, {
                        method: 'POST'
                    }).then(response => {
                        if (response.ok) {
                            // Remove alert from UI
                            location.reload(); // Simple refresh for now
                        }
                    });
                }

                // WebSocket event handlers
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    updateDashboard(data);
                };

                ws.onopen = function() {
                    console.log('Security monitoring WebSocket connected');
                };

                ws.onclose = function() {
                    console.log('Security monitoring WebSocket disconnected');
                    // Attempt to reconnect after 5 seconds
                    setTimeout(() => location.reload(), 5000);
                };

                // Initialize dashboard
                document.addEventListener('DOMContentLoaded', function() {
                    initCharts();

                    // Load initial dashboard data
                    fetch('/api/security/dashboard')
                        .then(response => response.json())
                        .then(data => {
                            updateMetrics(data.summary);

                            // Update alerts
                            if (data.active_alerts && data.active_alerts.length > 0) {
                                const alertsContainer = document.getElementById('alerts-container');
                                alertsContainer.innerHTML = '';
                                data.active_alerts.forEach(addAlert);
                            }
                        })
                        .catch(error => console.error('Error loading dashboard data:', error));
                });
            </script>
        </body>
        </html>
        """

        return HTMLResponse(content=html_content)

    return router
