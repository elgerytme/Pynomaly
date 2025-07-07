"""WebSocket endpoints for real-time updates."""

import json
import logging

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from pynomaly.application.services.task_tracking_service import (
    TaskInfo,
    TaskTrackingService,
)
from pynomaly.infrastructure.config import Container
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
    websocket: WebSocket, container: Container = Depends(get_container)
):
    """Main WebSocket endpoint for real-time updates."""
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
