"""WebSocket routes implementation using pure models."""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Optional
from uuid import uuid4

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..models.websocket_models import (
    ConnectionStatus,
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
        require_auth=False  # Simplified for demo
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
            "user_agent": websocket.headers.get("user-agent", "unknown")
        }
        
        # Register connection
        connection = websocket_service.register_connection(
            connection_id=connection_id,
            user_id=None,  # Would be extracted from auth in real implementation
            client_info=client_info
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
                    "timestamp": datetime.utcnow().isoformat()
                },
                timestamp=datetime.utcnow()
            )
            
            await websocket.send_text(json.dumps({
                "type": welcome_message.type.value,
                "data": welcome_message.data
            }))\n            \n            # Start heartbeat task
            heartbeat_task = asyncio.create_task(\n                send_heartbeat(websocket, connection_id, websocket_service)\n            )\n            \n            # Handle incoming messages\n            while True:\n                data = await websocket.receive_text()\n                \n                try:\n                    message_data = json.loads(data)\n                    await handle_websocket_message(\n                        websocket,\n                        connection_id,\n                        message_data,\n                        websocket_service\n                    )\n                except json.JSONDecodeError:\n                    logger.warning(f\"Invalid JSON received from {connection_id}\")\n                except Exception as e:\n                    logger.error(f\"Error handling message from {connection_id}: {e}\")\n                    \n        except WebSocketDisconnect:\n            logger.info(f\"WebSocket disconnected: {connection_id}\")\n        except Exception as e:\n            logger.error(f\"WebSocket error for {connection_id}: {e}\")\n        finally:\n            # Cleanup\n            websocket_service.unregister_connection(connection_id)\n            if 'heartbeat_task' in locals():\n                heartbeat_task.cancel()\n    \n    @router.websocket(\"/notifications/{user_id}\")\n    async def user_notifications(websocket: WebSocket, user_id: str):\n        \"\"\"WebSocket endpoint for user-specific notifications.\"\"\"\n        await websocket.accept()\n        \n        connection_id = str(uuid4())\n        \n        # Register connection for specific user\n        websocket_service.register_connection(\n            connection_id=connection_id,\n            user_id=user_id,\n            client_info={\"type\": \"notifications\"}\n        )\n        \n        # Subscribe to user notifications\n        websocket_service.subscribe_to_topic(\n            connection_id,\n            SubscriptionTopic.USER_NOTIFICATIONS\n        )\n        \n        logger.info(f\"User notification WebSocket connected: {user_id}\")\n        \n        try:\n            while True:\n                # Keep connection alive and handle messages\n                data = await websocket.receive_text()\n                # Handle user notification commands if needed\n                \n        except WebSocketDisconnect:\n            logger.info(f\"User notification WebSocket disconnected: {user_id}\")\n        finally:\n            websocket_service.unregister_connection(connection_id)\n    \n    @router.get(\"/metrics\")\n    async def get_websocket_metrics():\n        \"\"\"Get WebSocket service metrics.\"\"\"\n        metrics = websocket_service.get_metrics()\n        \n        return {\n            \"active_connections\": metrics.active_connections,\n            \"total_connections\": metrics.total_connections,\n            \"messages_sent\": metrics.messages_sent,\n            \"messages_received\": metrics.messages_received,\n            \"bytes_sent\": metrics.bytes_sent,\n            \"bytes_received\": metrics.bytes_received,\n            \"errors\": metrics.errors,\n            \"last_updated\": metrics.last_updated.isoformat()\n        }\n    \n    @router.get(\"/connections\")\n    async def get_active_connections():\n        \"\"\"Get list of active WebSocket connections.\"\"\"\n        connections = websocket_service.get_active_connections()\n        \n        return {\n            \"connections\": [\n                {\n                    \"connection_id\": conn.connection_id,\n                    \"user_id\": conn.user_id,\n                    \"connected_at\": conn.connected_at.isoformat(),\n                    \"last_activity\": conn.last_activity.isoformat(),\n                    \"status\": conn.status.value,\n                    \"subscriptions\": [topic.value for topic in conn.subscriptions]\n                }\n                for conn in connections\n            ]\n        }\n    \n    return router\n\n\nasync def send_heartbeat(\n    websocket: WebSocket,\n    connection_id: str,\n    service: WebSocketService\n) -> None:\n    \"\"\"Send periodic heartbeat messages.\n    \n    Args:\n        websocket: WebSocket connection\n        connection_id: Connection identifier\n        service: WebSocket service instance\n    \"\"\"\n    try:\n        while True:\n            await asyncio.sleep(service.config.heartbeat_interval)\n            \n            heartbeat_message = {\n                \"type\": MessageType.HEARTBEAT.value,\n                \"data\": {\n                    \"timestamp\": datetime.utcnow().isoformat(),\n                    \"connection_id\": connection_id\n                }\n            }\n            \n            await websocket.send_text(json.dumps(heartbeat_message))\n            \n    except asyncio.CancelledError:\n        logger.debug(f\"Heartbeat cancelled for {connection_id}\")\n    except Exception as e:\n        logger.error(f\"Heartbeat error for {connection_id}: {e}\")\n\n\nasync def handle_websocket_message(\n    websocket: WebSocket,\n    connection_id: str,\n    message_data: Dict,\n    service: WebSocketService\n) -> None:\n    \"\"\"Handle incoming WebSocket message.\n    \n    Args:\n        websocket: WebSocket connection\n        connection_id: Connection identifier\n        message_data: Parsed message data\n        service: WebSocket service instance\n    \"\"\"\n    message_type = message_data.get(\"type\")\n    data = message_data.get(\"data\", {})\n    \n    if message_type == \"subscribe\":\n        # Handle subscription request\n        topic_name = data.get(\"topic\")\n        if topic_name:\n            try:\n                topic = SubscriptionTopic(topic_name)\n                service.subscribe_to_topic(connection_id, topic)\n                \n                response = {\n                    \"type\": \"subscription_success\",\n                    \"data\": {\n                        \"topic\": topic_name,\n                        \"message\": f\"Subscribed to {topic_name}\"\n                    }\n                }\n                await websocket.send_text(json.dumps(response))\n                \n            except ValueError:\n                response = {\n                    \"type\": \"error\",\n                    \"data\": {\n                        \"message\": f\"Invalid topic: {topic_name}\"\n                    }\n                }\n                await websocket.send_text(json.dumps(response))\n    \n    elif message_type == \"unsubscribe\":\n        # Handle unsubscription request\n        topic_name = data.get(\"topic\")\n        if topic_name:\n            try:\n                topic = SubscriptionTopic(topic_name)\n                service.unsubscribe_from_topic(connection_id, topic)\n                \n                response = {\n                    \"type\": \"unsubscription_success\",\n                    \"data\": {\n                        \"topic\": topic_name,\n                        \"message\": f\"Unsubscribed from {topic_name}\"\n                    }\n                }\n                await websocket.send_text(json.dumps(response))\n                \n            except ValueError:\n                response = {\n                    \"type\": \"error\",\n                    \"data\": {\n                        \"message\": f\"Invalid topic: {topic_name}\"\n                    }\n                }\n                await websocket.send_text(json.dumps(response))\n    \n    elif message_type == \"ping\":\n        # Handle ping request\n        response = {\n            \"type\": \"pong\",\n            \"data\": {\n                \"timestamp\": datetime.utcnow().isoformat()\n            }\n        }\n        await websocket.send_text(json.dumps(response))\n    \n    else:\n        logger.warning(f\"Unknown message type: {message_type}\")
