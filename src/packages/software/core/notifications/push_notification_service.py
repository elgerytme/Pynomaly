"""Push Notification Service for Mobile Quality Monitoring.

Service for sending push notifications to mobile devices for
data quality alerts and incident management.
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend

logger = logging.getLogger(__name__)


class NotificationPlatform(Enum):
    """Supported notification platforms."""
    FCM = "fcm"  # Firebase Cloud Messaging
    APNS = "apns"  # Apple Push Notification Service
    WEB_PUSH = "web_push"  # Web Push Protocol


class NotificationPriority(Enum):
    """Notification priority levels."""
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


@dataclass
class PushSubscription:
    """Push notification subscription."""
    user_id: str
    platform: NotificationPlatform
    token: str
    endpoint: str = ""
    keys: Dict[str, str] = field(default_factory=dict)
    
    # User preferences
    enabled: bool = True
    quiet_hours_start: Optional[str] = None  # "22:00"
    quiet_hours_end: Optional[str] = None    # "08:00"
    severity_threshold: str = "medium"
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_used: datetime = field(default_factory=datetime.utcnow)
    device_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PushNotification:
    """Push notification data."""
    notification_id: str
    user_id: str
    title: str
    body: str
    
    # Platform-specific data
    data: Dict[str, Any] = field(default_factory=dict)
    
    # Notification options
    priority: NotificationPriority = NotificationPriority.NORMAL
    ttl: int = 3600  # Time to live in seconds
    collapse_key: Optional[str] = None
    
    # UI options
    icon: Optional[str] = None
    badge: Optional[str] = None
    sound: Optional[str] = None
    click_action: Optional[str] = None
    
    # Scheduling
    scheduled_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    
    # Tracking
    created_at: datetime = field(default_factory=datetime.utcnow)
    sent_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    clicked_at: Optional[datetime] = None
    
    def to_fcm_payload(self) -> Dict[str, Any]:
        """Convert to FCM payload format."""
        payload = {
            "notification": {
                "title": self.title,
                "body": self.body
            },
            "data": self.data
        }
        
        if self.icon:
            payload["notification"]["icon"] = self.icon
        
        if self.click_action:
            payload["notification"]["click_action"] = self.click_action
        
        if self.priority == NotificationPriority.HIGH:
            payload["android"] = {"priority": "high"}
            payload["apns"] = {
                "headers": {"apns-priority": "10"},
                "payload": {"aps": {"alert": {"title": self.title, "body": self.body}}}
            }
        
        return payload
    
    def to_apns_payload(self) -> Dict[str, Any]:
        """Convert to APNS payload format."""
        payload = {
            "aps": {
                "alert": {
                    "title": self.title,
                    "body": self.body
                }
            }
        }
        
        if self.badge:
            payload["aps"]["badge"] = self.badge
        
        if self.sound:
            payload["aps"]["sound"] = self.sound
        
        if self.priority == NotificationPriority.HIGH:
            payload["aps"]["priority"] = 10
        
        # Add custom data
        for key, value in self.data.items():
            payload[key] = value
        
        return payload
    
    def to_web_push_payload(self) -> Dict[str, Any]:
        """Convert to Web Push payload format."""
        payload = {
            "notification": {
                "title": self.title,
                "body": self.body,
                "tag": self.collapse_key or self.notification_id,
                "requireInteraction": self.priority == NotificationPriority.HIGH
            },
            "data": self.data
        }
        
        if self.icon:
            payload["notification"]["icon"] = self.icon
        
        if self.badge:
            payload["notification"]["badge"] = self.badge
        
        if self.click_action:
            payload["notification"]["actions"] = [
                {
                    "action": "view",
                    "title": "View",
                    "icon": self.icon
                }
            ]
        
        return payload


class PushNotificationService:
    """Service for sending push notifications to mobile devices."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize push notification service.
        
        Args:
            config: Service configuration
        """
        self.config = config or {}
        
        # Platform configurations
        self.fcm_config = self.config.get('fcm', {})
        self.apns_config = self.config.get('apns', {})
        self.web_push_config = self.config.get('web_push', {})
        
        # Storage
        self.subscriptions: Dict[str, List[PushSubscription]] = {}
        self.notification_history: Dict[str, PushNotification] = {}
        
        # Rate limiting
        self.rate_limits: Dict[str, List[datetime]] = {}
        
        logger.info("Push Notification Service initialized")
    
    async def subscribe(self, subscription: PushSubscription) -> bool:
        """Subscribe user to push notifications.
        
        Args:
            subscription: Push subscription data
            
        Returns:
            True if subscription successful
        """
        try:
            user_id = subscription.user_id
            
            if user_id not in self.subscriptions:
                self.subscriptions[user_id] = []
            
            # Remove existing subscription for same platform
            self.subscriptions[user_id] = [
                sub for sub in self.subscriptions[user_id] 
                if sub.platform != subscription.platform
            ]
            
            # Add new subscription
            self.subscriptions[user_id].append(subscription)
            
            logger.info(f"Push subscription added for user {user_id} on {subscription.platform.value}")
            return True
        
        except Exception as e:
            logger.error(f"Error subscribing user {subscription.user_id}: {e}")
            return False
    
    async def unsubscribe(self, user_id: str, platform: NotificationPlatform = None) -> bool:
        """Unsubscribe user from push notifications.
        
        Args:
            user_id: User identifier
            platform: Platform to unsubscribe from (None for all)
            
        Returns:
            True if unsubscription successful
        """
        try:
            if user_id not in self.subscriptions:
                return True
            
            if platform:
                # Remove subscription for specific platform
                self.subscriptions[user_id] = [
                    sub for sub in self.subscriptions[user_id] 
                    if sub.platform != platform
                ]
            else:
                # Remove all subscriptions
                self.subscriptions[user_id] = []
            
            logger.info(f"Push subscription removed for user {user_id}")
            return True
        
        except Exception as e:
            logger.error(f"Error unsubscribing user {user_id}: {e}")
            return False
    
    async def send_notification(self, notification: PushNotification) -> Dict[str, Any]:
        """Send push notification to user.
        
        Args:
            notification: Notification to send
            
        Returns:
            Send result
        """
        try:
            user_id = notification.user_id
            
            if user_id not in self.subscriptions:
                return {"success": False, "error": "No subscriptions found"}
            
            # Check rate limiting
            if not self._check_rate_limit(user_id):
                return {"success": False, "error": "Rate limit exceeded"}
            
            # Get active subscriptions
            active_subscriptions = [
                sub for sub in self.subscriptions[user_id] 
                if sub.enabled and self._is_within_allowed_hours(sub)
            ]
            
            if not active_subscriptions:
                return {"success": False, "error": "No active subscriptions"}
            
            # Send to all platforms
            results = []
            for subscription in active_subscriptions:
                try:
                    if subscription.platform == NotificationPlatform.FCM:
                        result = await self._send_fcm_notification(notification, subscription)
                    elif subscription.platform == NotificationPlatform.APNS:
                        result = await self._send_apns_notification(notification, subscription)
                    elif subscription.platform == NotificationPlatform.WEB_PUSH:
                        result = await self._send_web_push_notification(notification, subscription)
                    else:
                        result = {"success": False, "error": "Unsupported platform"}
                    
                    results.append({
                        "platform": subscription.platform.value,
                        "result": result
                    })
                    
                except Exception as e:
                    logger.error(f"Error sending to {subscription.platform.value}: {e}")
                    results.append({
                        "platform": subscription.platform.value,
                        "result": {"success": False, "error": str(e)}
                    })
            
            # Update notification tracking
            notification.sent_at = datetime.utcnow()
            self.notification_history[notification.notification_id] = notification
            
            # Update rate limiting
            self._update_rate_limit(user_id)
            
            success_count = sum(1 for r in results if r["result"]["success"])
            
            return {
                "success": success_count > 0,
                "results": results,
                "sent_count": success_count,
                "total_count": len(results)
            }
        
        except Exception as e:
            logger.error(f"Error sending notification {notification.notification_id}: {e}")
            return {"success": False, "error": str(e)}
    
    async def send_bulk_notifications(self, notifications: List[PushNotification]) -> List[Dict[str, Any]]:
        """Send bulk push notifications.
        
        Args:
            notifications: List of notifications to send
            
        Returns:
            List of send results
        """
        tasks = []
        for notification in notifications:
            task = asyncio.create_task(self.send_notification(notification))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "notification_id": notifications[i].notification_id,
                    "success": False,
                    "error": str(result)
                })
            else:
                processed_results.append({
                    "notification_id": notifications[i].notification_id,
                    **result
                })
        
        return processed_results
    
    async def _send_fcm_notification(self, notification: PushNotification, 
                                   subscription: PushSubscription) -> Dict[str, Any]:
        """Send FCM notification."""
        if not self.fcm_config.get('server_key'):
            return {"success": False, "error": "FCM server key not configured"}
        
        url = "https://fcm.googleapis.com/fcm/send"
        headers = {
            "Authorization": f"key={self.fcm_config['server_key']}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "to": subscription.token,
            **notification.to_fcm_payload(),
            "time_to_live": notification.ttl
        }
        
        if notification.collapse_key:
            payload["collapse_key"] = notification.collapse_key
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                result = await response.json()
                
                if response.status == 200 and result.get("success", 0) > 0:
                    return {"success": True, "message_id": result.get("results", [{}])[0].get("message_id")}
                else:
                    return {"success": False, "error": result.get("error", "Unknown FCM error")}
    
    async def _send_apns_notification(self, notification: PushNotification, 
                                    subscription: PushSubscription) -> Dict[str, Any]:
        """Send APNS notification."""
        if not self.apns_config.get('key_id') or not self.apns_config.get('team_id'):
            return {"success": False, "error": "APNS credentials not configured"}
        
        # Generate JWT token for APNS
        token = self._generate_apns_jwt()
        
        # APNS HTTP/2 endpoint
        url = f"https://api.push.apple.com/3/device/{subscription.token}"
        
        headers = {
            "Authorization": f"bearer {token}",
            "apns-topic": self.apns_config.get('bundle_id', 'com.example.app'),
            "apns-priority": "10" if notification.priority == NotificationPriority.HIGH else "5"
        }
        
        if notification.collapse_key:
            headers["apns-collapse-id"] = notification.collapse_key
        
        if notification.expires_at:
            headers["apns-expiration"] = str(int(notification.expires_at.timestamp()))
        
        payload = notification.to_apns_payload()
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    return {"success": True, "apns_id": response.headers.get("apns-id")}
                else:
                    error_data = await response.json() if response.content_type == "application/json" else {}
                    return {"success": False, "error": error_data.get("reason", f"APNS error: {response.status}")}
    
    async def _send_web_push_notification(self, notification: PushNotification, 
                                        subscription: PushSubscription) -> Dict[str, Any]:
        """Send Web Push notification."""
        if not self.web_push_config.get('vapid_public_key'):
            return {"success": False, "error": "Web Push VAPID keys not configured"}
        
        # Generate VAPID JWT
        vapid_token = self._generate_vapid_jwt(subscription.endpoint)
        
        headers = {
            "Authorization": f"vapid t={vapid_token}, k={self.web_push_config['vapid_public_key']}",
            "Content-Type": "application/json",
            "TTL": str(notification.ttl)
        }
        
        payload = notification.to_web_push_payload()
        
        async with aiohttp.ClientSession() as session:
            async with session.post(subscription.endpoint, headers=headers, json=payload) as response:
                if response.status in [200, 201, 204]:
                    return {"success": True}
                else:
                    return {"success": False, "error": f"Web Push error: {response.status}"}
    
    def _generate_apns_jwt(self) -> str:
        """Generate JWT token for APNS authentication."""
        now = datetime.utcnow()
        
        payload = {
            "iss": self.apns_config['team_id'],
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(hours=1)).timestamp())
        }
        
        headers = {
            "kid": self.apns_config['key_id'],
            "alg": "ES256"
        }
        
        # Load private key
        private_key = serialization.load_pem_private_key(
            self.apns_config['private_key'].encode(),
            password=None,
            backend=default_backend()
        )
        
        token = jwt.encode(payload, private_key, algorithm="ES256", headers=headers)
        return token
    
    def _generate_vapid_jwt(self, endpoint: str) -> str:
        """Generate VAPID JWT for Web Push."""
        now = datetime.utcnow()
        
        payload = {
            "aud": endpoint,
            "exp": int((now + timedelta(hours=12)).timestamp()),
            "sub": f"mailto:{self.web_push_config.get('contact_email', 'noreply@example.com')}"
        }
        
        # Load private key
        private_key = serialization.load_pem_private_key(
            self.web_push_config['vapid_private_key'].encode(),
            password=None,
            backend=default_backend()
        )
        
        token = jwt.encode(payload, private_key, algorithm="ES256")
        return token
    
    def _check_rate_limit(self, user_id: str) -> bool:
        """Check if user has exceeded rate limit."""
        if user_id not in self.rate_limits:
            return True
        
        # Remove old entries (older than 1 hour)
        cutoff_time = datetime.utcnow() - timedelta(hours=1)
        self.rate_limits[user_id] = [
            timestamp for timestamp in self.rate_limits[user_id] 
            if timestamp > cutoff_time
        ]
        
        # Check if under limit (max 100 per hour)
        return len(self.rate_limits[user_id]) < 100
    
    def _update_rate_limit(self, user_id: str):
        """Update rate limit counter for user."""
        if user_id not in self.rate_limits:
            self.rate_limits[user_id] = []
        
        self.rate_limits[user_id].append(datetime.utcnow())
    
    def _is_within_allowed_hours(self, subscription: PushSubscription) -> bool:
        """Check if current time is within allowed hours for user."""
        if not subscription.quiet_hours_start or not subscription.quiet_hours_end:
            return True
        
        now = datetime.utcnow().time()
        start_time = datetime.strptime(subscription.quiet_hours_start, "%H:%M").time()
        end_time = datetime.strptime(subscription.quiet_hours_end, "%H:%M").time()
        
        if start_time <= end_time:
            # Same day (e.g., 08:00 to 22:00)
            return not (start_time <= now <= end_time)
        else:
            # Across midnight (e.g., 22:00 to 08:00)
            return not (now >= start_time or now <= end_time)
    
    async def get_subscription_info(self, user_id: str) -> Dict[str, Any]:
        """Get subscription info for user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Subscription information
        """
        subscriptions = self.subscriptions.get(user_id, [])
        
        return {
            "user_id": user_id,
            "subscription_count": len(subscriptions),
            "platforms": [sub.platform.value for sub in subscriptions],
            "enabled_count": len([sub for sub in subscriptions if sub.enabled]),
            "subscriptions": [
                {
                    "platform": sub.platform.value,
                    "enabled": sub.enabled,
                    "created_at": sub.created_at.isoformat(),
                    "last_used": sub.last_used.isoformat(),
                    "quiet_hours": {
                        "start": sub.quiet_hours_start,
                        "end": sub.quiet_hours_end
                    } if sub.quiet_hours_start else None
                }
                for sub in subscriptions
            ]
        }
    
    async def get_notification_stats(self, user_id: str = None) -> Dict[str, Any]:
        """Get notification statistics.
        
        Args:
            user_id: Optional user to filter by
            
        Returns:
            Notification statistics
        """
        notifications = list(self.notification_history.values())
        
        if user_id:
            notifications = [n for n in notifications if n.user_id == user_id]
        
        now = datetime.utcnow()
        last_24h = now - timedelta(hours=24)
        last_7d = now - timedelta(days=7)
        
        stats = {
            "total_notifications": len(notifications),
            "sent_notifications": len([n for n in notifications if n.sent_at]),
            "delivered_notifications": len([n for n in notifications if n.delivered_at]),
            "clicked_notifications": len([n for n in notifications if n.clicked_at]),
            "last_24h": len([n for n in notifications if n.created_at > last_24h]),
            "last_7d": len([n for n in notifications if n.created_at > last_7d]),
            "by_priority": {
                "high": len([n for n in notifications if n.priority == NotificationPriority.HIGH]),
                "normal": len([n for n in notifications if n.priority == NotificationPriority.NORMAL]),
                "low": len([n for n in notifications if n.priority == NotificationPriority.LOW])
            }
        }
        
        if user_id:
            stats["user_id"] = user_id
        
        return stats
    
    async def cleanup_old_notifications(self, days: int = 30):
        """Clean up old notifications.
        
        Args:
            days: Number of days to keep notifications
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        to_remove = [
            notification_id for notification_id, notification in self.notification_history.items()
            if notification.created_at < cutoff_date
        ]
        
        for notification_id in to_remove:
            del self.notification_history[notification_id]
        
        logger.info(f"Cleaned up {len(to_remove)} old notifications")
    
    async def test_notification(self, user_id: str, platform: NotificationPlatform = None) -> Dict[str, Any]:
        """Send test notification to user.
        
        Args:
            user_id: User identifier
            platform: Optional platform to test
            
        Returns:
            Test result
        """
        test_notification = PushNotification(
            notification_id=f"test_{user_id}_{datetime.utcnow().timestamp()}",
            user_id=user_id,
            title="Test Notification",
            body="This is a test notification from Software Quality Monitoring",
            priority=NotificationPriority.NORMAL,
            data={"type": "test", "timestamp": datetime.utcnow().isoformat()}
        )
        
        return await self.send_notification(test_notification)