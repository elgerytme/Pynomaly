"""WebSocket routes implementation using pure models."""

import asyncio
import json
import logging
from datetime import datetime
from uuid import uuid4

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..models.websocket_models import (
    MessageType,
    SubscriptionTopic,
    WebSocketConfig,
    WebSocketMessageEnvelope,
)
from ..services.websocket_service import WebSocketService

logger = logging.getLogger(__name__)


def create_websocket_router() -> APIRouter:
    """Create WebSocket router with pure model dependencies.

    Returns:
        Configured WebSocket router
    """
    router = APIRouter()

    # Create WebSocket service with default config
    config = WebSocketConfig(
        heartbeat_interval=30,
        connection_timeout=300,
        max_connections_per_user=5,
        require_auth=False,  # Simplified for demo
    )

    websocket_service = WebSocketService(config)

    @router.websocket("/connect")
    async def websocket_endpoint(websocket: WebSocket):
        """Main WebSocket endpoint for real-time communication."""
        await websocket.accept()

        # Generate connection ID
        connection_id = str(uuid4())

        # Extract client info
        client_info = {
            "client_host": websocket.client.host if websocket.client else "unknown",
            "user_agent": websocket.headers.get("user-agent", "unknown"),
        }

        # Register connection
        connection = websocket_service.register_connection(
            connection_id=connection_id,
            user_id=None,  # Would be extracted from auth in a real implementation
            client_info=client_info,
        )

        logger.info(f"WebSocket connection established: {connection_id}")

        try:
            # Send welcome message
            welcome_message = WebSocketMessageEnvelope(
                message_id=str(uuid4()),
                type=MessageType.CONNECT,
                topic=None,
                data={
                    "message": "Connected successfully",
                    "connection_id": connection_id,
                    "timestamp": datetime.utcnow().isoformat(),
                },
                timestamp=datetime.utcnow(),
            )

            await websocket.send_text(
                json.dumps(
                    {"type": welcome_message.type.value, "data": welcome_message.data}
                )
            )

            # Start heartbeat task
            heartbeat_task = asyncio.create_task(
                send_heartbeat(websocket, connection_id, websocket_service)
            )

            # Handle incoming messages
            while True:
                data = await websocket.receive_text()

                try:
                    message_data = json.loads(data)
                    await handle_websocket_message(
                        websocket, connection_id, message_data, websocket_service
                    )
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON received from {connection_id}")
                except Exception as e:
                    logger.error(f"Error handling message from {connection_id}: {e}")

        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected: {connection_id}")
        except Exception as e:
            logger.error(f"WebSocket error for {connection_id}: {e}")
        finally:
            # Cleanup
            websocket_service.unregister_connection(connection_id)
            if "heartbeat_task" in locals():
                heartbeat_task.cancel()

    @router.websocket("/notifications/{user_id}")
    async def user_notifications(websocket: WebSocket, user_id: str):
        """WebSocket endpoint for user-specific notifications."""
        await websocket.accept()

        connection_id = str(uuid4())

        # Register connection for specific user
        websocket_service.register_connection(
            connection_id=connection_id,
            user_id=user_id,
            client_info={"type": "notifications"},
        )

        # Subscribe to user notifications
        websocket_service.subscribe_to_topic(
            connection_id, SubscriptionTopic.USER_NOTIFICATIONS
        )

        logger.info(f"User notification WebSocket connected: {user_id}")

        try:
            while True:
                # Keep connection alive and handle messages
                data = await websocket.receive_text()
                # Handle user notification commands if needed
        except WebSocketDisconnect:
            logger.info(f"User notification WebSocket disconnected: {user_id}")
        finally:
            websocket_service.unregister_connection(connection_id)

    @router.get("/metrics")
    async def get_websocket_metrics():
        """Get WebSocket service metrics."""
        metrics = websocket_service.get_metrics()

        return {
            "active_connections": metrics.active_connections,
            "total_connections": metrics.total_connections,
            "messages_sent": metrics.messages_sent,
            "messages_received": metrics.messages_received,
            "bytes_sent": metrics.bytes_sent,
            "bytes_received": metrics.bytes_received,
            "errors": metrics.errors,
            "last_updated": metrics.last_updated.isoformat(),
        }

    @router.get("/connections")
    async def get_active_connections():
        """Get list of active WebSocket connections."""
        connections = websocket_service.get_active_connections()

        return {
            "connections": [
                {
                    "connection_id": conn.connection_id,
                    "user_id": conn.user_id,
                    "connected_at": conn.connected_at.isoformat(),
                    "last_activity": conn.last_activity.isoformat(),
                    "status": conn.status.value,
                    "subscriptions": [topic.value for topic in conn.subscriptions],
                }
                for conn in connections
            ]
        }

    return router


async def send_heartbeat(
    websocket: WebSocket, connection_id: str, service: WebSocketService
) -> None:
    """Send periodic heartbeat messages.

    Args:
        websocket: WebSocket connection
        connection_id: Connection identifier
        service: WebSocket service instance
    """
    try:
        while True:
            await asyncio.sleep(service.config.heartbeat_interval)

            heartbeat_message = {
                "type": MessageType.HEARTBEAT.value,
                "data": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "connection_id": connection_id,
                },
            }

            await websocket.send_text(json.dumps(heartbeat_message))

    except asyncio.CancelledError:
        logger.debug(f"Heartbeat cancelled for {connection_id}")
    except Exception as e:
        logger.error(f"Heartbeat error for {connection_id}: {e}")


async def handle_websocket_message(
    websocket: WebSocket,
    connection_id: str,
    message_data: dict,
    service: WebSocketService,
) -> None:
    """Handle incoming WebSocket message.

    Args:
        websocket: WebSocket connection
        connection_id: Connection identifier
        message_data: Parsed message data
        service: WebSocket service instance
    """
    message_type = message_data.get("type")
    data = message_data.get("data", {})

    if message_type == "subscribe":
        # Handle subscription request
        topic_name = data.get("topic")
        if topic_name:
            try:
                topic = SubscriptionTopic(topic_name)
                service.subscribe_to_topic(connection_id, topic)

                response = {
                    "type": "subscription_success",
                    "data": {
                        "topic": topic_name,
                        "message": f"Subscribed to {topic_name}",
                    },
                }
                await websocket.send_text(json.dumps(response))

            except ValueError:
                response = {
                    "type": "error",
                    "data": {"message": f"Invalid topic: {topic_name}"},
                }
                await websocket.send_text(json.dumps(response))

    elif message_type == "unsubscribe":
        # Handle unsubscription request
        topic_name = data.get("topic")
        if topic_name:
            try:
                topic = SubscriptionTopic(topic_name)
                service.unsubscribe_from_topic(connection_id, topic)

                response = {
                    "type": "unsubscription_success",
                    "data": {
                        "topic": topic_name,
                        "message": f"Unsubscribed from {topic_name}",
                    },
                }
                await websocket.send_text(json.dumps(response))

            except ValueError:
                response = {
                    "type": "error",
                    "data": {"message": f"Invalid topic: {topic_name}"},
                }
                await websocket.send_text(json.dumps(response))

    elif message_type == "ping":
        # Handle ping request
        response = {
            "type": "pong",
            "data": {"timestamp": datetime.utcnow().isoformat()},
        }
        await websocket.send_text(json.dumps(response))

    else:
        logger.warning(f"Unknown message type: {message_type}")
