"""
Push Notifications Router

Handles PWA push notification subscriptions and message sending
for real-time anomaly processing alerts and system notifications.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any
from urllib.parse import urlparse

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field, validator
from pywebpush import WebPushException, webpush
from sqlalchemy.orm import Session

from ....domain.entities.user import User
from ....infrastructure.config.settings import get_settings
from ....infrastructure.database.connection import get_db
from ....infrastructure.monitoring.metrics import track_event
from ....infrastructure.security.auth import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/push", tags=["push-notifications"])
security = HTTPBearer()
settings = get_settings()


class PushSubscription(BaseModel):
    """Push subscription data from browser"""

    endpoint: str = Field(..., description="Push service endpoint URL")
    keys: dict[str, str] = Field(..., description="Encryption keys (p256dh, auth)")

    @validator("endpoint")
    def validate_endpoint(cls, v):
        try:
            parsed = urlparse(v)
            if not parsed.scheme or not parsed.netloc:
                raise ValueError("Invalid endpoint URL")
            return v
        except Exception as e:
            raise ValueError("Invalid endpoint URL format") from e

    @validator("keys")
    def validate_keys(cls, v):
        required_keys = {"p256dh", "auth"}
        if not all(key in v for key in required_keys):
            raise ValueError(f"Keys must contain: {required_keys}")
        return v


class SubscriptionRequest(BaseModel):
    """Request to register push subscription"""

    subscription: PushSubscription
    userAgent: str | None = Field(None, description="Browser user agent")
    timestamp: int | None = Field(None, description="Registration timestamp")
    deviceInfo: dict[str, Any] | None = Field(None, description="Device information")


class UnsubscriptionRequest(BaseModel):
    """Request to remove push subscription"""

    endpoint: str = Field(..., description="Push service endpoint to remove")


class PushMessage(BaseModel):
    """Push notification message"""

    title: str = Field(..., max_length=100, description="Notification title")
    body: str = Field(..., max_length=300, description="Notification body")
    icon: str | None = Field(
        "/static/icons/icon-192x192.png", description="Notification icon"
    )
    badge: str | None = Field(
        "/static/icons/badge-72x72.png", description="Notification badge"
    )
    data: dict[str, Any] | None = Field(None, description="Custom notification data")
    url: str | None = Field(None, description="URL to open on click")
    actions: list[dict[str, str]] | None = Field(
        None, description="Notification actions"
    )
    priority: str | None = Field("normal", description="Priority: low, normal, high")
    requireInteraction: bool | None = Field(
        False, description="Require user interaction"
    )
    silent: bool | None = Field(False, description="Silent notification")
    vibrate: list[int] | None = Field([200, 100, 200], description="Vibration pattern")
    ttl: int | None = Field(86400, description="Time to live in seconds")


class BroadcastMessage(BaseModel):
    """Broadcast message to multiple users"""

    message: PushMessage
    userIds: list[int] | None = Field(None, description="Specific user IDs to send to")
    tags: list[str] | None = Field(None, description="User tags to filter by")
    sendToAll: bool | None = Field(False, description="Send to all subscribed users")


# In-memory subscription storage (replace with proper database in production)
# This should be moved to a proper database processor
push_subscriptions: dict[int, list[dict[str, Any]]] = {}


@router.post("/subscribe", status_code=status.HTTP_201_CREATED)
async def subscribe_to_push(
    request: SubscriptionRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Register a new push notification subscription for the current user.

    This endpoint stores the push subscription details from the user's browser
    and enables the server to send push notifications to that specific device.
    """
    try:
        user_id = current_user.id
        subscription_data = {
            "endpoint": request.subscription.endpoint,
            "keys": request.subscription.keys,
            "userAgent": request.userAgent,
            "timestamp": request.timestamp or int(datetime.now().timestamp()),
            "deviceInfo": request.deviceInfo,
            "created": datetime.now().isoformat(),
            "lastUsed": datetime.now().isoformat(),
            "active": True,
        }

        # Initialize user subscriptions if not exists
        if user_id not in push_subscriptions:
            push_subscriptions[user_id] = []

        # Check if subscription already exists (by endpoint)
        existing_subscriptions = push_subscriptions[user_id]
        for i, sub in enumerate(existing_subscriptions):
            if sub["endpoint"] == request.subscription.endpoint:
                # Update existing subscription
                existing_subscriptions[i] = subscription_data
                logger.info(f"Updated push subscription for user {user_id}")
                break
        else:
            # Add new subscription
            existing_subscriptions.append(subscription_data)
            logger.info(f"Added new push subscription for user {user_id}")

        # Track subscription event
        background_tasks.add_task(
            track_event,
            "push_subscription_registered",
            {
                "user_id": user_id,
                "endpoint_domain": urlparse(request.subscription.endpoint).netloc,
                "user_agent": request.userAgent[:100] if request.userAgent else None,
            },
        )

        # Send welcome notification
        if settings.security.push_welcome_notification:
            background_tasks.add_task(
                send_welcome_notification, user_id, subscription_data
            )

        return {
            "success": True,
            "message": "Push subscription registered successfully",
            "subscriptionId": len(existing_subscriptions) - 1,
        }

    except Exception as e:
        logger.error(f"Failed to register push subscription: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to register push subscription: {str(e)}",
        ) from e


@router.post("/unsubscribe", status_code=status.HTTP_200_OK)
async def unsubscribe_from_push(
    request: UnsubscriptionRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Remove a push notification subscription for the current user.

    This endpoint removes the specified subscription endpoint from the user's
    registered subscriptions, preventing future notifications to that device.
    """
    try:
        user_id = current_user.id

        if user_id not in push_subscriptions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No subscriptions found for user",
            )

        # Find and remove subscription by endpoint
        user_subs = push_subscriptions[user_id]
        original_count = len(user_subs)

        push_subscriptions[user_id] = [
            sub for sub in user_subs if sub["endpoint"] != request.endpoint
        ]

        removed_count = original_count - len(push_subscriptions[user_id])

        if removed_count == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Subscription endpoint not found",
            )

        logger.info(f"Removed {removed_count} push subscription(s) for user {user_id}")

        # Track unsubscription event
        background_tasks.add_task(
            track_event,
            "push_subscription_removed",
            {
                "user_id": user_id,
                "endpoint_domain": urlparse(request.endpoint).netloc,
                "removed_count": removed_count,
            },
        )

        return {
            "success": True,
            "message": f"Successfully removed {removed_count} subscription(s)",
            "removedCount": removed_count,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to unsubscribe from push: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to unsubscribe from push: {str(e)}",
        ) from e


@router.post("/send", status_code=status.HTTP_200_OK)
async def send_push_notification(
    user_id: int,
    message: PushMessage,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Send a push notification to a specific user.

    Requires admin privileges or sending to self.
    """
    try:
        # Check permissions
        if current_user.id != user_id and not current_user.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to send notification to this user",
            )

        if user_id not in push_subscriptions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No active subscriptions found for user",
            )

        user_subs = push_subscriptions[user_id]
        active_subs = [sub for sub in user_subs if sub.get("active", True)]

        if not active_subs:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No active subscriptions found for user",
            )

        # Send notification to all user's devices
        background_tasks.add_task(
            send_to_subscriptions, active_subs, message.dict(), current_user.id
        )

        return {
            "success": True,
            "message": f"Notification queued for {len(active_subs)} device(s)",
            "deviceCount": len(active_subs),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to send push notification: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to send push notification: {str(e)}",
        ) from e


@router.post("/broadcast", status_code=status.HTTP_200_OK)
async def broadcast_push_notification(
    broadcast: BroadcastMessage,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Broadcast a push notification to multiple users.

    Requires admin privileges.
    """
    try:
        # Check admin permissions
        if not current_user.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin privileges required for broadcast notifications",
            )

        target_users = []

        if broadcast.sendToAll:
            target_users = list(push_subscriptions.keys())
        elif broadcast.userIds:
            target_users = broadcast.userIds
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Must specify either userIds or sendToAll=true",
            )

        total_devices = 0
        all_subscriptions = []

        for user_id in target_users:
            if user_id in push_subscriptions:
                user_subs = push_subscriptions[user_id]
                active_subs = [sub for sub in user_subs if sub.get("active", True)]
                all_subscriptions.extend(active_subs)
                total_devices += len(active_subs)

        if not all_subscriptions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No active subscriptions found for target users",
            )

        # Send broadcast notification
        background_tasks.add_task(
            send_to_subscriptions,
            all_subscriptions,
            broadcast.message.dict(),
            current_user.id,
            is_broadcast=True,
        )

        return {
            "success": True,
            "message": (
                f"Broadcast notification queued for {total_devices} device(s) "
                f"across {len(target_users)} user(s)"
            ),
            "deviceCount": total_devices,
            "userCount": len(target_users),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to broadcast push notification: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to broadcast push notification: {str(e)}",
        ) from e


@router.get("/subscriptions", status_code=status.HTTP_200_OK)
async def get_user_subscriptions(
    current_user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    """
    Get all push subscriptions for the current user.
    """
    try:
        user_id = current_user.id

        if user_id not in push_subscriptions:
            return {"subscriptions": [], "count": 0}

        user_subs = push_subscriptions[user_id]

        # Remove sensitive keys from response
        safe_subscriptions = []
        for sub in user_subs:
            safe_sub = {
                "endpoint": sub["endpoint"],
                "created": sub["created"],
                "lastUsed": sub["lastUsed"],
                "active": sub.get("active", True),
                "userAgent": sub.get("userAgent"),
                "deviceInfo": sub.get("deviceInfo"),
            }
            safe_subscriptions.append(safe_sub)

        return {"subscriptions": safe_subscriptions, "count": len(safe_subscriptions)}

    except Exception as e:
        logger.error(f"Failed to get user subscriptions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get user subscriptions: {str(e)}",
        ) from e


@router.delete("/subscriptions", status_code=status.HTTP_200_OK)
async def clear_all_subscriptions(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Remove all push subscriptions for the current user.
    """
    try:
        user_id = current_user.id

        if user_id not in push_subscriptions:
            return {
                "success": True,
                "message": "No subscriptions found to remove",
                "removedCount": 0,
            }

        removed_count = len(push_subscriptions[user_id])
        del push_subscriptions[user_id]

        logger.info(
            f"Removed all {removed_count} push subscriptions for user {user_id}"
        )

        # Track unsubscription event
        background_tasks.add_task(
            track_event,
            "push_subscriptions_cleared",
            {"user_id": user_id, "removed_count": removed_count},
        )

        return {
            "success": True,
            "message": f"Successfully removed all {removed_count} subscription(s)",
            "removedCount": removed_count,
        }

    except Exception as e:
        logger.error(f"Failed to clear subscriptions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear subscriptions: {str(e)}",
        ) from e


# Background task functions


async def send_to_subscriptions(
    subscriptions: list[dict[str, Any]],
    message_data: dict[str, Any],
    sender_id: int,
    is_broadcast: bool = False,
):
    """Send push notification to list of subscriptions"""

    if (
        not settings.security.vapid_private_key
        or not settings.security.vapid_public_key
    ):
        logger.error("VAPID keys not configured - cannot send push notifications")
        return

    # Prepare notification payload
    notification_payload = json.dumps(
        {
            "title": message_data["title"],
            "body": message_data["body"],
            "icon": message_data.get("icon", "/static/icons/icon-192x192.png"),
            "badge": message_data.get("badge", "/static/icons/badge-72x72.png"),
            "data": message_data.get("data", {}),
            "url": message_data.get("url"),
            "actions": message_data.get("actions", []),
            "requireInteraction": message_data.get("requireInteraction", False),
            "silent": message_data.get("silent", False),
            "vibrate": message_data.get("vibrate", [200, 100, 200]),
            "timestamp": int(datetime.now().timestamp()),
        }
    )

    # Send to each subscription
    success_count = 0
    failure_count = 0

    for subscription in subscriptions:
        try:
            # Prepare subscription info for pywebpush
            subscription_info = {
                "endpoint": subscription["endpoint"],
                "keys": subscription["keys"],
            }

            # Send push notification
            webpush(
                subscription_info=subscription_info,
                data=notification_payload,
                vapid_private_key=settings.security.vapid_private_key,
                vapid_claims={
                    "sub": f"mailto:{settings.security.vapid_email}",
                    "exp": int((datetime.now() + timedelta(hours=12)).timestamp()),
                },
                timeout=10,
                ttl=message_data.get("ttl", settings.security.push_notification_ttl),
            )

            success_count += 1

            # Update last used timestamp
            subscription["lastUsed"] = datetime.now().isoformat()

            logger.debug(
                f"Push notification sent successfully to {subscription['endpoint'][:50]}..."
            )

        except WebPushException as e:
            failure_count += 1

            # Handle different error types
            if e.response and e.response.status_code in [400, 404, 410]:
                # Subscription is invalid, mark as inactive
                subscription["active"] = False
                logger.warning(f"Push subscription invalid, marked inactive: {e}")
            else:
                logger.error(f"Push notification failed: {e}")

        except Exception as e:
            failure_count += 1
            logger.error(f"Unexpected error sending push notification: {e}")

    # Track sending measurements
    await track_event(
        "push_notification_sent",
        {
            "sender_id": sender_id,
            "is_broadcast": is_broadcast,
            "success_count": success_count,
            "failure_count": failure_count,
            "total_devices": len(subscriptions),
            "title": message_data["title"][:50],
        },
    )

    logger.info(
        f"Push notification delivery complete: {success_count} success, {failure_count} failed"
    )


async def send_welcome_notification(user_id: int, subscription: dict[str, Any]):
    """Send welcome notification to new subscription"""

    welcome_message = {
        "title": "🎉 Welcome to Software!",
        "body": "Push notifications are now enabled. You'll receive alerts about anomaly processing results and system updates.",
        "icon": "/static/icons/icon-192x192.png",
        "badge": "/static/icons/badge-72x72.png",
        "data": {"type": "welcome", "url": "/dashboard"},
        "url": "/dashboard",
        "requireInteraction": False,
        "silent": False,
    }

    await send_to_subscriptions([subscription], welcome_message, user_id)


# Utility functions for anomaly processing integration


async def send_anomaly_alert(user_id: int, anomaly_data: dict[str, Any]):
    """Send push notification for anomaly processing results"""

    if user_id not in push_subscriptions:
        return

    anomaly_count = anomaly_data.get("anomaly_count", 0)
    data_collection_name = anomaly_data.get("data_collection_name", "Unknown DataCollection")
    severity = anomaly_data.get("severity", "medium")

    # Determine priority based on severity
    priority = "high" if severity == "high" else "normal"
    require_interaction = severity == "high"

    # Create notification message
    message = {
        "title": f"🚨 {anomaly_count} Anomalies Detected",
        "body": f"Found {anomaly_count} anomalies in {data_collection_name}. Click to view details.",
        "icon": "/static/icons/alert-icon.png",
        "badge": "/static/icons/alert-badge.png",
        "data": {
            "type": "anomaly_alert",
            "anomaly_count": anomaly_count,
            "data_collection_name": data_collection_name,
            "severity": severity,
            "processing_id": anomaly_data.get("processing_id"),
            "url": f"/results/{anomaly_data.get('processing_id')}",
        },
        "url": f"/results/{anomaly_data.get('processing_id')}",
        "actions": [
            {"action": "view", "title": "View Results"},
            {"action": "dismiss", "title": "Dismiss"},
        ],
        "requireInteraction": require_interaction,
        "silent": False,
        "vibrate": [300, 100, 300, 100, 300] if severity == "high" else [200, 100, 200],
    }

    user_subs = push_subscriptions.get(user_id, [])
    active_subs = [sub for sub in user_subs if sub.get("active", True)]

    if active_subs:
        await send_to_subscriptions(active_subs, message, user_id)


async def send_system_notification(
    user_ids: list[int], title: str, body: str, data: dict[str, Any] = None
):
    """Send system notification to multiple users"""

    message = {
        "title": title,
        "body": body,
        "icon": "/static/icons/icon-192x192.png",
        "badge": "/static/icons/badge-72x72.png",
        "data": data or {},
        "requireInteraction": False,
        "silent": False,
    }

    all_subscriptions = []
    for user_id in user_ids:
        if user_id in push_subscriptions:
            user_subs = push_subscriptions[user_id]
            active_subs = [sub for sub in user_subs if sub.get("active", True)]
            all_subscriptions.extend(active_subs)

    if all_subscriptions:
        await send_to_subscriptions(all_subscriptions, message, 0, is_broadcast=True)
