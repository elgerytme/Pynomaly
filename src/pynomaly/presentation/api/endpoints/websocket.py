"""WebSocket endpoints for real-time updates."""

import asyncio
import json
import logging
from datetime import datetime

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from pynomaly.application.services.task_tracking_service import (
    TaskInfo,
    TaskTrackingService,
)
from pynomaly.infrastructure.config import Container
from pynomaly.infrastructure.auth import get_auth, create_websocket_auth_dependency
from pynomaly.presentation.api.deps import get_container

router = APIRouter()
logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self):
        # Active connections by task_id
        self.task_connections: dict[str, set[WebSocket]] = {}
        # Global connections (receive all updates)
        self.global_connections: set[WebSocket] = set()
        # Connection to task mappings
        self.connection_tasks: dict[WebSocket, set[str]] = {}

    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.global_connections.add(websocket)
        self.connection_tasks[websocket] = set()
        logger.info(f"WebSocket connected: {websocket.client}")

    def disconnect(self, websocket: WebSocket):
        """Handle WebSocket disconnection."""
        # Remove from global connections
        self.global_connections.discard(websocket)

        # Remove from task-specific connections
        if websocket in self.connection_tasks:
            for task_id in self.connection_tasks[websocket]:
                if task_id in self.task_connections:
                    self.task_connections[task_id].discard(websocket)
                    if not self.task_connections[task_id]:
                        del self.task_connections[task_id]
            del self.connection_tasks[websocket]

        logger.info(f"WebSocket disconnected: {websocket.client}")

    def subscribe_to_task(self, websocket: WebSocket, task_id: str):
        """Subscribe a connection to a specific task."""
        if task_id not in self.task_connections:
            self.task_connections[task_id] = set()

        self.task_connections[task_id].add(websocket)
        self.connection_tasks[websocket].add(task_id)
        logger.info(f"WebSocket subscribed to task {task_id}")

    async def send_task_update(self, task_info: TaskInfo):
        """Send task update to subscribed connections."""
        message = {"type": "task_update", "data": task_info.to_dict()}

        # Send to task-specific subscribers
        task_id = task_info.task_id
        if task_id in self.task_connections:
            disconnected = set()
            for websocket in self.task_connections[task_id]:
                try:
                    await websocket.send_text(json.dumps(message))
                except Exception as e:
                    logger.error(f"Error sending task update: {e}")
                    disconnected.add(websocket)

            # Clean up disconnected clients
            for websocket in disconnected:
                self.disconnect(websocket)

    async def send_global_update(self, message: dict):
        """Send update to all global connections."""
        disconnected = set()
        for websocket in self.global_connections:
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending global update: {e}")
                disconnected.add(websocket)

        # Clean up disconnected clients
        for websocket in disconnected:
            self.disconnect(websocket)


# Global connection manager instance
connection_manager = ConnectionManager()


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket, 
    container: Container = Depends(get_container)
):
    """Main WebSocket endpoint for real-time updates."""
    # Authenticate the WebSocket connection
    try:
        auth_service = get_auth()
        ws_auth_dep = create_websocket_auth_dependency(auth_service)
        user = await ws_auth_dep(websocket)
        logger.info(f"Authenticated WebSocket user: {user.username}")
    except Exception as e:
        logger.warning(f"WebSocket authentication failed: {e}")
        await websocket.close(code=4001, reason="Authentication required")
        return
    
    await connection_manager.connect(websocket)

    try:
        while True:
            # Wait for client messages
            data = await websocket.receive_text()

            try:
                message = json.loads(data)
                await handle_websocket_message(websocket, message, container)
            except json.JSONDecodeError:
                await websocket.send_text(
                    json.dumps({"type": "error", "message": "Invalid JSON message"})
                )
            except Exception as e:
                logger.error(f"Error handling WebSocket message: {e}")
                await websocket.send_text(
                    json.dumps({"type": "error", "message": str(e)})
                )

    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)


async def handle_websocket_message(
    websocket: WebSocket, message: dict, container: Container
):
    """Handle incoming WebSocket messages."""
    message_type = message.get("type")

    if message_type == "subscribe_task":
        # Subscribe to specific task updates
        task_id = message.get("task_id")
        if task_id:
            connection_manager.subscribe_to_task(websocket, task_id)

            # Send current task status
            task_service = container.task_tracking_service()
            task_info = task_service.get_task(task_id)
            if task_info:
                await websocket.send_text(
                    json.dumps({"type": "task_status", "data": task_info.to_dict()})
                )

    elif message_type == "get_active_tasks":
        # Send list of active tasks
        task_service = container.task_tracking_service()
        active_tasks = task_service.get_active_tasks()

        await websocket.send_text(
            json.dumps(
                {
                    "type": "active_tasks",
                    "data": [task.to_dict() for task in active_tasks],
                }
            )
        )

    elif message_type == "get_recent_tasks":
        # Send list of recent tasks
        task_service = container.task_tracking_service()
        limit = message.get("limit", 20)
        recent_tasks = task_service.get_recent_tasks(limit)

        await websocket.send_text(
            json.dumps(
                {
                    "type": "recent_tasks",
                    "data": [task.to_dict() for task in recent_tasks],
                }
            )
        )

    elif message_type == "cancel_task":
        # Cancel a task
        task_id = message.get("task_id")
        if task_id:
            task_service = container.task_tracking_service()
            success = task_service.cancel_task(task_id)

            await websocket.send_text(
                json.dumps(
                    {"type": "task_cancelled", "task_id": task_id, "success": success}
                )
            )


def setup_task_tracking_websockets(task_service: TaskTrackingService):
    """Set up WebSocket notifications for task tracking service."""

    def task_update_callback(task_info: TaskInfo):
        """Callback for task updates to send via WebSocket."""
        import asyncio

        # Create a task to send the update
        asyncio.create_task(connection_manager.send_task_update(task_info))

    # This would need to be called when tasks are created
    # to subscribe the WebSocket callback to task updates
    return task_update_callback


@router.websocket("/ws/detections")
async def detections_websocket(
    websocket: WebSocket,
    container: Container = Depends(get_container)
):
    """WebSocket endpoint for real-time anomaly detection results."""
    # Authenticate the WebSocket connection
    try:
        auth_service = get_auth()
        ws_auth_dep = create_websocket_auth_dependency(auth_service)
        user = await ws_auth_dep(websocket)
        logger.info(f"Authenticated WebSocket user for detections: {user.username}")
    except Exception as e:
        logger.warning(f"WebSocket authentication failed: {e}")
        await websocket.close(code=4001, reason="Authentication required")
        return

    await websocket.accept()
    
    try:
        # Get streaming detection service
        streaming_service = container.streaming_detection_service()
        
        # Subscribe to anomaly detection results
        detection_queue = asyncio.Queue()
        
        def detection_callback(result):
            """Callback for new detection results."""
            asyncio.create_task(detection_queue.put(result))
        
        # Register callback with streaming service
        streaming_service.register_detection_callback(detection_callback)
        
        # Send initial connection confirmation
        await websocket.send_text(json.dumps({
            "type": "connected",
            "message": "Connected to real-time detection stream",
            "timestamp": datetime.utcnow().isoformat(),
            "user": user.username
        }))
        
        # Main message handling loop
        async def handle_incoming_messages():
            """Handle incoming WebSocket messages."""
            while True:
                try:
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    
                    if message.get("type") == "ping":
                        await websocket.send_text(json.dumps({
                            "type": "pong",
                            "timestamp": datetime.utcnow().isoformat()
                        }))
                    elif message.get("type") == "start_stream":
                        # Start streaming detection if not already running
                        await streaming_service.start_stream()
                        await websocket.send_text(json.dumps({
                            "type": "stream_started",
                            "message": "Real-time detection stream started",
                            "timestamp": datetime.utcnow().isoformat()
                        }))
                    elif message.get("type") == "stop_stream":
                        # Stop streaming detection
                        await streaming_service.stop_stream()
                        await websocket.send_text(json.dumps({
                            "type": "stream_stopped",
                            "message": "Real-time detection stream stopped",
                            "timestamp": datetime.utcnow().isoformat()
                        }))
                        
                except json.JSONDecodeError:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "Invalid JSON message",
                        "timestamp": datetime.utcnow().isoformat()
                    }))
                except Exception as e:
                    logger.error(f"Error handling WebSocket message: {e}")
                    break
        
        # Start message handler
        message_handler = asyncio.create_task(handle_incoming_messages())
        
        # Main detection result forwarding loop
        while True:
            try:
                # Wait for detection results or timeout
                try:
                    result = await asyncio.wait_for(detection_queue.get(), timeout=1.0)
                    
                    # Send detection result to client
                    await websocket.send_text(json.dumps({
                        "type": "anomaly_detection",
                        "data": {
                            "batch_id": result.batch_id,
                            "anomalies_detected": result.anomalies_detected,
                            "batch_size": result.batch_size,
                            "processing_time_ms": result.processing_time_ms,
                            "timestamp": result.timestamp.isoformat(),
                            "sample_results": [
                                {
                                    "id": str(sr.id),
                                    "is_anomaly": sr.is_anomaly,
                                    "scores": [s.value for s in sr.scores] if sr.scores else [],
                                    "anomaly_threshold": sr.anomaly_threshold,
                                    "execution_time_ms": sr.execution_time_ms,
                                    "metadata": sr.metadata
                                } for sr in result.sample_results[:10]  # Limit to first 10 for performance
                            ]
                        },
                        "timestamp": datetime.utcnow().isoformat()
                    }))
                    
                except asyncio.TimeoutError:
                    # Send keepalive ping
                    await websocket.send_text(json.dumps({
                        "type": "keepalive",
                        "timestamp": datetime.utcnow().isoformat()
                    }))
                    
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error in detection WebSocket loop: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"Error in detection stream: {str(e)}",
                    "timestamp": datetime.utcnow().isoformat()
                }))
                break
        
        # Cleanup
        if not message_handler.done():
            message_handler.cancel()
        
        # Unregister callback
        streaming_service.unregister_detection_callback(detection_callback)
        
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for user: {user.username if 'user' in locals() else 'unknown'}")
    except Exception as e:
        logger.error(f"Error in detections WebSocket: {e}")
    finally:
        try:
            await websocket.close()
        except:
            pass


@router.get("/tasks/{task_id}/sse")
async def task_sse_endpoint(
    task_id: str, container: Container = Depends(get_container)
):
    """Server-Sent Events endpoint for task updates (alternative to WebSocket)."""
    import asyncio

    from fastapi.responses import StreamingResponse

    async def event_generator():
        task_service = container.task_tracking_service()

        # Send initial task status
        task_info = task_service.get_task(task_id)
        if task_info:
            yield f"data: {json.dumps(task_info.to_dict())}\n\n"

        # Set up callback for updates
        updates_queue = asyncio.Queue()

        def callback(task_info: TaskInfo):
            asyncio.create_task(updates_queue.put(task_info))

        task_service.subscribe_to_task(task_id, callback)

        try:
            while True:
                # Wait for updates with timeout
                try:
                    task_info = await asyncio.wait_for(
                        updates_queue.get(), timeout=30.0
                    )
                    yield f"data: {json.dumps(task_info.to_dict())}\n\n"

                    # Stop if task is completed
                    if task_info.status.value in ["completed", "failed", "cancelled"]:
                        break

                except TimeoutError:
                    # Send keepalive
                    yield 'data: {"type": "keepalive"}\n\n'

        finally:
            task_service.unsubscribe_from_task(task_id, callback)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
