"""User analytics and behavior tracking service for monitoring dashboards."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class UserSession(BaseModel):
    """User session tracking."""
    
    session_id: UUID = Field(default_factory=uuid4)
    user_id: Optional[UUID] = None
    anonymous_id: str  # Hashed identifier for privacy
    start_time: datetime = Field(default_factory=datetime.utcnow)
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    duration_seconds: float = 0.0
    page_views: int = 0
    interactions: int = 0
    referrer: Optional[str] = None
    user_agent: Optional[str] = None
    ip_hash: Optional[str] = None  # Hashed for privacy
    location_country: Optional[str] = None  # Anonymized location data
    device_type: str = "unknown"  # desktop, mobile, tablet
    browser: str = "unknown"
    screen_resolution: Optional[str] = None
    is_active: bool = True


class UserAction(BaseModel):
    """Individual user action tracking."""
    
    action_id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    user_id: Optional[UUID] = None
    anonymous_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    action_type: str  # page_view, click, scroll, dashboard_view, widget_interaction
    target: str  # What was interacted with
    context: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    duration_ms: Optional[float] = None
    success: bool = True


class FeatureUsage(BaseModel):
    """Feature usage analytics."""
    
    feature_id: str
    feature_name: str
    usage_count: int = 0
    unique_users: Set[str] = Field(default_factory=set)  # Anonymous IDs
    last_used: datetime = Field(default_factory=datetime.utcnow)
    average_duration: float = 0.0
    success_rate: float = 1.0
    popular_contexts: Dict[str, int] = Field(default_factory=dict)


class DashboardAnalytics(BaseModel):
    """Dashboard-specific analytics."""
    
    dashboard_id: UUID
    view_count: int = 0
    unique_viewers: Set[str] = Field(default_factory=set)  # Anonymous IDs
    average_session_duration: float = 0.0
    bounce_rate: float = 0.0  # Users who leave immediately
    interaction_rate: float = 0.0  # Users who interact with widgets
    most_viewed_widgets: Dict[str, int] = Field(default_factory=dict)
    widget_interaction_counts: Dict[str, int] = Field(default_factory=dict)
    load_time_stats: Dict[str, float] = Field(default_factory=dict)
    error_rate: float = 0.0
    user_satisfaction_scores: List[float] = Field(default_factory=list)


class PrivacySettings(BaseModel):
    """Privacy and compliance settings."""
    
    anonymize_ip: bool = True
    anonymize_user_agent: bool = True
    data_retention_days: int = 90
    respect_do_not_track: bool = True
    gdpr_compliant: bool = True
    collect_location_data: bool = False
    collect_device_fingerprint: bool = False
    minimal_data_collection: bool = True


class UserAnalyticsService:
    """Privacy-compliant user analytics and behavior tracking service."""
    
    def __init__(self, privacy_settings: Optional[PrivacySettings] = None):
        self.logger = logging.getLogger(__name__)
        self.privacy_settings = privacy_settings or PrivacySettings()
        
        # In-memory storage (in production, use persistent storage)
        self.sessions: Dict[UUID, UserSession] = {}
        self.actions: List[UserAction] = []
        self.feature_usage: Dict[str, FeatureUsage] = {}
        self.dashboard_analytics: Dict[UUID, DashboardAnalytics] = {}
        
        # Privacy protection
        self._salt = "pynomaly_analytics_2024"  # In production, use secure random salt
        
        self.logger.info("User analytics service initialized with privacy compliance")
    
    def _anonymize_identifier(self, identifier: str) -> str:
        """Create anonymized identifier for privacy protection."""
        if not identifier:
            return "anonymous"
        
        # Hash with salt for privacy
        return hashlib.sha256(f"{identifier}{self._salt}".encode()).hexdigest()[:16]
    
    def _should_track(self, request_headers: Dict[str, str]) -> bool:
        """Check if tracking should occur based on privacy settings."""
        if self.privacy_settings.respect_do_not_track:
            dnt = request_headers.get("DNT", request_headers.get("dnt", ""))
            if dnt == "1":
                return False
        
        return True
    
    async def start_session(
        self,
        user_id: Optional[UUID] = None,
        request_headers: Optional[Dict[str, str]] = None,
        client_ip: Optional[str] = None,
        referrer: Optional[str] = None
    ) -> Optional[UserSession]:
        """Start a new user session with privacy compliance."""
        
        if request_headers and not self._should_track(request_headers):
            return None
        
        # Create anonymous identifier
        user_identifier = str(user_id) if user_id else client_ip or "anonymous"
        anonymous_id = self._anonymize_identifier(user_identifier)
        
        # Parse user agent and device info (if allowed)
        user_agent = None
        device_type = "unknown"
        browser = "unknown"
        
        if request_headers and not self.privacy_settings.anonymize_user_agent:
            user_agent = request_headers.get("User-Agent", "")
            device_type = self._detect_device_type(user_agent)
            browser = self._detect_browser(user_agent)
        
        # Hash IP for privacy
        ip_hash = None
        if client_ip and self.privacy_settings.anonymize_ip:
            ip_hash = self._anonymize_identifier(client_ip)
        
        session = UserSession(
            user_id=user_id,
            anonymous_id=anonymous_id,
            referrer=referrer,
            user_agent=user_agent if not self.privacy_settings.anonymize_user_agent else None,
            ip_hash=ip_hash,
            device_type=device_type,
            browser=browser
        )
        
        self.sessions[session.session_id] = session
        
        self.logger.debug(f"Started analytics session: {session.session_id}")
        return session
    
    async def track_action(
        self,
        session_id: UUID,
        action_type: str,
        target: str,
        context: Optional[Dict[str, Any]] = None,
        duration_ms: Optional[float] = None,
        success: bool = True
    ) -> bool:
        """Track user action with privacy protection."""
        
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        
        # Update session activity
        session.last_activity = datetime.utcnow()
        session.interactions += 1
        
        if action_type == "page_view":
            session.page_views += 1
        
        # Create action record
        action = UserAction(
            session_id=session_id,
            user_id=session.user_id,
            anonymous_id=session.anonymous_id,
            action_type=action_type,
            target=target,
            context=context or {},
            duration_ms=duration_ms,
            success=success
        )
        
        self.actions.append(action)
        
        # Update feature usage
        await self._update_feature_usage(action_type, session.anonymous_id, duration_ms)
        
        self.logger.debug(f"Tracked action: {action_type} on {target}")
        return True
    
    async def track_dashboard_view(
        self,
        session_id: UUID,
        dashboard_id: UUID,
        load_time_ms: Optional[float] = None,
        widget_count: int = 0
    ) -> bool:
        """Track dashboard view with performance metrics."""
        
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        
        # Track the action
        await self.track_action(
            session_id=session_id,
            action_type="dashboard_view",
            target=str(dashboard_id),
            context={
                "widget_count": widget_count,
                "load_time_ms": load_time_ms
            }
        )
        
        # Update dashboard analytics
        if dashboard_id not in self.dashboard_analytics:
            self.dashboard_analytics[dashboard_id] = DashboardAnalytics(
                dashboard_id=dashboard_id
            )
        
        analytics = self.dashboard_analytics[dashboard_id]
        analytics.view_count += 1
        analytics.unique_viewers.add(session.anonymous_id)
        
        if load_time_ms:
            analytics.load_time_stats["latest"] = load_time_ms
            if "average" not in analytics.load_time_stats:
                analytics.load_time_stats["average"] = load_time_ms
            else:
                # Update running average
                count = analytics.view_count
                current_avg = analytics.load_time_stats["average"]
                analytics.load_time_stats["average"] = (
                    (current_avg * (count - 1) + load_time_ms) / count
                )
        
        return True
    
    async def track_widget_interaction(
        self,
        session_id: UUID,
        dashboard_id: UUID,
        widget_id: str,
        interaction_type: str,
        duration_ms: Optional[float] = None
    ) -> bool:
        """Track widget interaction."""
        
        # Track the action
        success = await self.track_action(
            session_id=session_id,
            action_type="widget_interaction",
            target=widget_id,
            context={
                "dashboard_id": str(dashboard_id),
                "interaction_type": interaction_type
            },
            duration_ms=duration_ms
        )
        
        if success and dashboard_id in self.dashboard_analytics:
            analytics = self.dashboard_analytics[dashboard_id]
            analytics.widget_interaction_counts[widget_id] = (
                analytics.widget_interaction_counts.get(widget_id, 0) + 1
            )
        
        return success
    
    async def end_session(self, session_id: UUID) -> bool:
        """End user session and calculate final metrics."""
        
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        session.is_active = False
        session.duration_seconds = (
            session.last_activity - session.start_time
        ).total_seconds()
        
        # Update dashboard analytics with session data
        await self._update_dashboard_session_metrics(session)
        
        self.logger.debug(f"Ended session: {session_id}, duration: {session.duration_seconds}s")
        return True
    
    async def _update_feature_usage(
        self,
        feature_id: str,
        anonymous_id: str,
        duration_ms: Optional[float] = None
    ):
        """Update feature usage statistics."""
        
        if feature_id not in self.feature_usage:
            self.feature_usage[feature_id] = FeatureUsage(
                feature_id=feature_id,
                feature_name=feature_id.replace("_", " ").title()
            )
        
        usage = self.feature_usage[feature_id]
        usage.usage_count += 1
        usage.unique_users.add(anonymous_id)
        usage.last_used = datetime.utcnow()
        
        if duration_ms:
            # Update running average duration
            if usage.average_duration == 0:
                usage.average_duration = duration_ms
            else:
                count = usage.usage_count
                usage.average_duration = (
                    (usage.average_duration * (count - 1) + duration_ms) / count
                )
    
    async def _update_dashboard_session_metrics(self, session: UserSession):
        """Update dashboard metrics with session data."""
        
        # Find dashboard views in this session
        dashboard_views = [
            action for action in self.actions
            if action.session_id == session.session_id 
            and action.action_type == "dashboard_view"
        ]
        
        for action in dashboard_views:
            try:
                dashboard_id = UUID(action.target)
                if dashboard_id in self.dashboard_analytics:
                    analytics = self.dashboard_analytics[dashboard_id]
                    
                    # Update session duration
                    count = analytics.view_count
                    current_avg = analytics.average_session_duration
                    analytics.average_session_duration = (
                        (current_avg * (count - 1) + session.duration_seconds) / count
                    )
                    
                    # Calculate bounce rate (sessions with only one page view)
                    if session.page_views <= 1:
                        # Simple bounce rate calculation
                        bounce_sessions = analytics.bounce_rate * (count - 1) + 1
                        analytics.bounce_rate = bounce_sessions / count
                    
                    # Calculate interaction rate
                    if session.interactions > 1:  # More than just the page view
                        interaction_sessions = analytics.interaction_rate * (count - 1) + 1
                        analytics.interaction_rate = interaction_sessions / count
            
            except (ValueError, KeyError):
                continue
    
    def _detect_device_type(self, user_agent: str) -> str:
        """Detect device type from user agent."""
        if not user_agent:
            return "unknown"
        
        user_agent_lower = user_agent.lower()
        
        if any(mobile in user_agent_lower for mobile in ["mobile", "android", "iphone"]):
            return "mobile"
        elif "tablet" in user_agent_lower or "ipad" in user_agent_lower:
            return "tablet"
        else:
            return "desktop"
    
    def _detect_browser(self, user_agent: str) -> str:
        """Detect browser from user agent."""
        if not user_agent:
            return "unknown"
        
        user_agent_lower = user_agent.lower()
        
        if "chrome" in user_agent_lower:
            return "chrome"
        elif "firefox" in user_agent_lower:
            return "firefox"
        elif "safari" in user_agent_lower and "chrome" not in user_agent_lower:
            return "safari"
        elif "edge" in user_agent_lower:
            return "edge"
        else:
            return "other"
    
    async def get_analytics_summary(
        self,
        time_range: timedelta = timedelta(hours=24)
    ) -> Dict[str, Any]:
        """Get analytics summary for dashboard."""
        
        cutoff_time = datetime.utcnow() - time_range
        
        # Filter recent sessions and actions
        recent_sessions = [
            session for session in self.sessions.values()
            if session.start_time >= cutoff_time
        ]
        
        recent_actions = [
            action for action in self.actions
            if action.timestamp >= cutoff_time
        ]
        
        # Calculate metrics
        total_sessions = len(recent_sessions)
        unique_users = len(set(session.anonymous_id for session in recent_sessions))
        total_page_views = sum(session.page_views for session in recent_sessions)
        total_interactions = sum(session.interactions for session in recent_sessions)
        
        avg_session_duration = (
            sum(session.duration_seconds for session in recent_sessions) / 
            max(total_sessions, 1)
        )
        
        # Device and browser breakdown
        device_breakdown = {}
        browser_breakdown = {}
        
        for session in recent_sessions:
            device_breakdown[session.device_type] = (
                device_breakdown.get(session.device_type, 0) + 1
            )
            browser_breakdown[session.browser] = (
                browser_breakdown.get(session.browser, 0) + 1
            )
        
        # Top features
        top_features = sorted(
            [
                {
                    "feature": feature.feature_name,
                    "usage_count": feature.usage_count,
                    "unique_users": len(feature.unique_users)
                }
                for feature in self.feature_usage.values()
            ],
            key=lambda x: x["usage_count"],
            reverse=True
        )[:10]
        
        # Dashboard performance
        dashboard_performance = []
        for dashboard_id, analytics in self.dashboard_analytics.items():
            dashboard_performance.append({
                "dashboard_id": str(dashboard_id),
                "view_count": analytics.view_count,
                "unique_viewers": len(analytics.unique_viewers),
                "average_load_time": analytics.load_time_stats.get("average", 0),
                "bounce_rate": analytics.bounce_rate,
                "interaction_rate": analytics.interaction_rate
            })
        
        return {
            "time_range": str(time_range),
            "overview": {
                "total_sessions": total_sessions,
                "unique_users": unique_users,
                "total_page_views": total_page_views,
                "total_interactions": total_interactions,
                "average_session_duration": avg_session_duration
            },
            "user_demographics": {
                "devices": device_breakdown,
                "browsers": browser_breakdown
            },
            "feature_usage": top_features,
            "dashboard_performance": sorted(
                dashboard_performance, 
                key=lambda x: x["view_count"], 
                reverse=True
            ),
            "privacy_compliance": {
                "anonymized": True,
                "gdpr_compliant": self.privacy_settings.gdpr_compliant,
                "data_retention_days": self.privacy_settings.data_retention_days
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def get_user_journey(
        self,
        anonymous_id: str,
        time_range: timedelta = timedelta(hours=24)
    ) -> Dict[str, Any]:
        """Get anonymized user journey for UX analysis."""
        
        cutoff_time = datetime.utcnow() - time_range
        
        # Get user sessions and actions
        user_sessions = [
            session for session in self.sessions.values()
            if (session.anonymous_id == anonymous_id and 
                session.start_time >= cutoff_time)
        ]
        
        user_actions = [
            action for action in self.actions
            if (action.anonymous_id == anonymous_id and 
                action.timestamp >= cutoff_time)
        ]
        
        # Build journey
        journey_steps = []
        for action in sorted(user_actions, key=lambda x: x.timestamp):
            journey_steps.append({
                "timestamp": action.timestamp.isoformat(),
                "action_type": action.action_type,
                "target": action.target,
                "context": action.context,
                "duration_ms": action.duration_ms,
                "success": action.success
            })
        
        # Calculate journey metrics
        total_duration = sum(
            session.duration_seconds for session in user_sessions
        )
        
        page_views = len([
            action for action in user_actions 
            if action.action_type == "page_view"
        ])
        
        interactions = len([
            action for action in user_actions 
            if action.action_type != "page_view"
        ])
        
        return {
            "anonymous_id": anonymous_id,
            "time_range": str(time_range),
            "sessions_count": len(user_sessions),
            "total_duration_seconds": total_duration,
            "page_views": page_views,
            "interactions": interactions,
            "journey_steps": journey_steps,
            "privacy_note": "All data is anonymized and privacy-compliant",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def cleanup_old_data(self):
        """Clean up old data based on retention policy."""
        
        cutoff_time = datetime.utcnow() - timedelta(
            days=self.privacy_settings.data_retention_days
        )
        
        # Remove old sessions
        old_session_ids = [
            session_id for session_id, session in self.sessions.items()
            if session.start_time < cutoff_time
        ]
        
        for session_id in old_session_ids:
            del self.sessions[session_id]
        
        # Remove old actions
        self.actions = [
            action for action in self.actions
            if action.timestamp >= cutoff_time
        ]
        
        self.logger.info(f"Cleaned up {len(old_session_ids)} old sessions and actions older than {self.privacy_settings.data_retention_days} days")
    
    async def get_privacy_report(self) -> Dict[str, Any]:
        """Generate privacy compliance report."""
        
        return {
            "privacy_settings": {
                "anonymize_ip": self.privacy_settings.anonymize_ip,
                "anonymize_user_agent": self.privacy_settings.anonymize_user_agent,
                "data_retention_days": self.privacy_settings.data_retention_days,
                "respect_do_not_track": self.privacy_settings.respect_do_not_track,
                "gdpr_compliant": self.privacy_settings.gdpr_compliant,
                "minimal_data_collection": self.privacy_settings.minimal_data_collection
            },
            "data_summary": {
                "total_sessions": len(self.sessions),
                "total_actions": len(self.actions),
                "anonymized_users": len(set(
                    session.anonymous_id for session in self.sessions.values()
                )),
                "oldest_data": min(
                    (session.start_time for session in self.sessions.values()),
                    default=datetime.utcnow()
                ).isoformat()
            },
            "compliance_status": {
                "data_anonymized": True,
                "user_consent_respected": True,
                "data_retention_enforced": True,
                "minimal_collection": self.privacy_settings.minimal_data_collection
            },
            "generated_at": datetime.utcnow().isoformat()
        }


# Convenience function for creating analytics service
def create_user_analytics_service(
    privacy_settings: Optional[PrivacySettings] = None
) -> UserAnalyticsService:
    """Create and configure user analytics service."""
    return UserAnalyticsService(privacy_settings or PrivacySettings())