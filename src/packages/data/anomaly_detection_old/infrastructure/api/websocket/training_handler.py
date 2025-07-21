"""WebSocket handler for real-time training monitoring.

Provides real-time updates for:
- Training progress and status
- Hyperparameter optimization trials
- Resource usage monitoring
- Training completion notifications
- Error and warning messages
"""

from __future__ import annotations

import json
import logging
from typing import Any
from uuid import UUID

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ValidationError

from interfaces.application.services.automated_training_service import (
    AutomatedTrainingService,
    TrainingConfig,
    TriggerType,
)
from interfaces.application.services.automl_service import OptimizationObjective

logger = logging.getLogger(__name__)


class TrainingMessage(BaseModel):
    """Base class for training WebSocket messages."""

    type: str
    training_id: str | None = None


class StartTrainingMessage(TrainingMessage):
    """Message to start a new training."""

    type: str = "start_training"
    detector_id: str
    dataset_id: str
    experiment_name: str | None = None
    enable_automl: bool = True
    optimization_objective: str = "auc"
    max_algorithms: int = 3
    enable_ensemble: bool = True
    max_optimization_time: int = 3600
    validation_split: float = 0.2


class CancelTrainingMessage(TrainingMessage):
    """Message to cancel an active training."""

    type: str = "cancel_training"
    training_id: str


class GetTrainingStatusMessage(TrainingMessage):
    """Message to get training status."""

    type: str = "get_training_status"
    training_id: str


class GetActiveTrainingsMessage(TrainingMessage):
    """Message to get all active trainings."""

    type: str = "get_active_trainings"


class GetTrainingHistoryMessage(TrainingMessage):
    """Message to get training history."""

    type: str = "get_training_history"
    detector_id: str | None = None
    limit: int = 50


class SubscribeTrainingUpdatesMessage(TrainingMessage):
    """Message to subscribe to training updates."""

    type: str = "subscribe_training_updates"
    training_id: str | None = None  # If None, subscribe to all trainings


class TrainingWebSocketHandler:
    """WebSocket handler for training operations."""

    def __init__(self, training_service: AutomatedTrainingService):
        """Initialize the WebSocket handler.

        Args:
            training_service: Automated training service
        """
        self.training_service = training_service
        self.connected_clients: dict[str, WebSocket] = {}
        self.client_subscriptions: dict[
            str, set[str]
        ] = {}  # client_id -> set of training_ids

    async def connect(self, websocket: WebSocket, client_id: str):
        """Handle WebSocket connection.

        Args:
            websocket: WebSocket connection
            client_id: Unique client identifier
        """
        await websocket.accept()
        self.connected_clients[client_id] = websocket
        self.client_subscriptions[client_id] = set()

        logger.info(f"Training WebSocket client {client_id} connected")

        # Send initial status
        await self._send_message(
            websocket,
            {
                "type": "connection_established",
                "client_id": client_id,
                "message": "Connected to training service",
            },
        )

    async def disconnect(self, client_id: str):
        """Handle WebSocket disconnection.

        Args:
            client_id: Client identifier
        """
        if client_id in self.connected_clients:
            del self.connected_clients[client_id]

        if client_id in self.client_subscriptions:
            del self.client_subscriptions[client_id]

        logger.info(f"Training WebSocket client {client_id} disconnected")

    async def handle_message(
        self, websocket: WebSocket, client_id: str, message: dict[str, Any]
    ):
        """Handle incoming WebSocket message.

        Args:
            websocket: WebSocket connection
            client_id: Client identifier
            message: Received message
        """
        try:
            message_type = message.get("type")

            if message_type == "start_training":
                await self._handle_start_training(websocket, client_id, message)
            elif message_type == "cancel_training":
                await self._handle_cancel_training(websocket, client_id, message)
            elif message_type == "get_training_status":
                await self._handle_get_training_status(websocket, client_id, message)
            elif message_type == "get_active_trainings":
                await self._handle_get_active_trainings(websocket, client_id, message)
            elif message_type == "get_training_history":
                await self._handle_get_training_history(websocket, client_id, message)
            elif message_type == "subscribe_training_updates":
                await self._handle_subscribe_training_updates(
                    websocket, client_id, message
                )
            else:
                await self._send_error(
                    websocket, f"Unknown message type: {message_type}"
                )

        except ValidationError as e:
            await self._send_error(websocket, f"Invalid message format: {str(e)}")
        except Exception as e:
            logger.error(f"Error handling training WebSocket message: {str(e)}")
            await self._send_error(websocket, f"Internal error: {str(e)}")

    async def _handle_start_training(
        self, websocket: WebSocket, client_id: str, message: dict[str, Any]
    ):
        """Handle start training message."""
        try:
            msg = StartTrainingMessage(**message)

            # Parse optimization objective
            objective_map = {
                "auc": OptimizationObjective.AUC,
                "precision": OptimizationObjective.PRECISION,
                "recall": OptimizationObjective.RECALL,
                "f1_score": OptimizationObjective.F1_SCORE,
                "balanced_accuracy": OptimizationObjective.BALANCED_ACCURACY,
                "detection_rate": OptimizationObjective.DETECTION_RATE,
            }

            objective = objective_map.get(
                msg.optimization_objective.lower(), OptimizationObjective.AUC
            )

            # Create training configuration
            config = TrainingConfig(
                detector_id=UUID(msg.detector_id),
                dataset_id=msg.dataset_id,
                experiment_name=msg.experiment_name,
                enable_automl=msg.enable_automl,
                optimization_objective=objective,
                max_algorithms=msg.max_algorithms,
                enable_ensemble=msg.enable_ensemble,
                max_optimization_time=msg.max_optimization_time,
                validation_split=msg.validation_split,
            )

            # Start training
            training_id = await self.training_service.schedule_training(
                config, TriggerType.MANUAL
            )

            # Subscribe client to this training
            self.client_subscriptions[client_id].add(training_id)

            await self._send_message(
                websocket,
                {
                    "type": "training_started",
                    "training_id": training_id,
                    "message": "Training started successfully",
                },
            )

        except ValueError as e:
            await self._send_error(websocket, f"Invalid detector ID: {str(e)}")
        except Exception as e:
            await self._send_error(websocket, f"Failed to start training: {str(e)}")

    async def _handle_cancel_training(
        self, websocket: WebSocket, client_id: str, message: dict[str, Any]
    ):
        """Handle cancel training message."""
        try:
            msg = CancelTrainingMessage(**message)

            success = await self.training_service.cancel_training(msg.training_id)

            if success:
                await self._send_message(
                    websocket,
                    {
                        "type": "training_cancelled",
                        "training_id": msg.training_id,
                        "message": "Training cancelled successfully",
                    },
                )
            else:
                await self._send_error(
                    websocket, f"Failed to cancel training {msg.training_id}"
                )

        except Exception as e:
            await self._send_error(websocket, f"Failed to cancel training: {str(e)}")

    async def _handle_get_training_status(
        self, websocket: WebSocket, client_id: str, message: dict[str, Any]
    ):
        """Handle get training status message."""
        try:
            msg = GetTrainingStatusMessage(**message)

            status = await self.training_service.get_training_status(msg.training_id)

            if status:
                await self._send_message(
                    websocket,
                    {
                        "type": "training_status",
                        "training_id": msg.training_id,
                        "status": status.to_dict(),
                    },
                )
            else:
                await self._send_error(
                    websocket, f"Training {msg.training_id} not found"
                )

        except Exception as e:
            await self._send_error(
                websocket, f"Failed to get training status: {str(e)}"
            )

    async def _handle_get_active_trainings(
        self, websocket: WebSocket, client_id: str, message: dict[str, Any]
    ):
        """Handle get active trainings message."""
        try:
            active_trainings = await self.training_service.get_active_trainings()

            await self._send_message(
                websocket,
                {
                    "type": "active_trainings",
                    "trainings": [training.to_dict() for training in active_trainings],
                },
            )

        except Exception as e:
            await self._send_error(
                websocket, f"Failed to get active trainings: {str(e)}"
            )

    async def _handle_get_training_history(
        self, websocket: WebSocket, client_id: str, message: dict[str, Any]
    ):
        """Handle get training history message."""
        try:
            msg = GetTrainingHistoryMessage(**message)

            detector_id = UUID(msg.detector_id) if msg.detector_id else None
            history = await self.training_service.get_training_history(
                detector_id, msg.limit
            )

            # Convert to dict format
            history_data = []
            for result in history:
                data = {
                    "training_id": result.training_id,
                    "detector_id": str(result.detector_id),
                    "status": result.status.value,
                    "trigger_type": result.trigger_type.value,
                    "best_algorithm": result.best_algorithm,
                    "best_score": result.best_score,
                    "training_time_seconds": result.training_time_seconds,
                    "trials_completed": result.trials_completed,
                    "model_version": result.model_version,
                    "performance_improvement": result.performance_improvement,
                    "dataset_id": result.dataset_id,
                    "experiment_name": result.experiment_name,
                    "start_time": (
                        result.start_time.isoformat() if result.start_time else None
                    ),
                    "completion_time": (
                        result.completion_time.isoformat()
                        if result.completion_time
                        else None
                    ),
                    "error_message": result.error_message,
                    "warnings": result.warnings,
                }
                history_data.append(data)

            await self._send_message(
                websocket, {"type": "training_history", "history": history_data}
            )

        except ValueError as e:
            await self._send_error(websocket, f"Invalid detector ID: {str(e)}")
        except Exception as e:
            await self._send_error(
                websocket, f"Failed to get training history: {str(e)}"
            )

    async def _handle_subscribe_training_updates(
        self, websocket: WebSocket, client_id: str, message: dict[str, Any]
    ):
        """Handle subscribe to training updates message."""
        try:
            msg = SubscribeTrainingUpdatesMessage(**message)

            if msg.training_id:
                self.client_subscriptions[client_id].add(msg.training_id)
                await self._send_message(
                    websocket,
                    {
                        "type": "subscribed",
                        "training_id": msg.training_id,
                        "message": f"Subscribed to training {msg.training_id} updates",
                    },
                )
            else:
                # Subscribe to all trainings
                self.client_subscriptions[client_id].add("*")
                await self._send_message(
                    websocket,
                    {
                        "type": "subscribed",
                        "message": "Subscribed to all training updates",
                    },
                )

        except Exception as e:
            await self._send_error(
                websocket, f"Failed to subscribe to updates: {str(e)}"
            )

    async def broadcast_training_update(
        self, training_id: str, update_data: dict[str, Any]
    ):
        """Broadcast training update to subscribed clients.

        Args:
            training_id: ID of the training
            update_data: Update data to broadcast
        """
        message = {
            "type": "training_update",
            "training_id": training_id,
            "data": update_data,
        }

        # Find clients subscribed to this training or all trainings
        subscribed_clients = []
        for client_id, subscriptions in self.client_subscriptions.items():
            if training_id in subscriptions or "*" in subscriptions:
                if client_id in self.connected_clients:
                    subscribed_clients.append(client_id)

        # Broadcast to subscribed clients
        for client_id in subscribed_clients:
            websocket = self.connected_clients[client_id]
            try:
                await self._send_message(websocket, message)
            except Exception as e:
                logger.error(
                    f"Failed to send training update to client {client_id}: {str(e)}"
                )
                # Remove disconnected client
                await self.disconnect(client_id)

    async def _send_message(self, websocket: WebSocket, message: dict[str, Any]):
        """Send message to WebSocket client.

        Args:
            websocket: WebSocket connection
            message: Message to send
        """
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Failed to send WebSocket message: {str(e)}")
            raise

    async def _send_error(self, websocket: WebSocket, error_message: str):
        """Send error message to WebSocket client.

        Args:
            websocket: WebSocket connection
            error_message: Error message
        """
        await self._send_message(
            websocket,
            {
                "type": "error",
                "message": error_message,
                "timestamp": "timestamp would be here",
            },
        )


# Global training handler instance
training_handler: TrainingWebSocketHandler | None = None


def get_training_handler() -> TrainingWebSocketHandler:
    """Get the global training WebSocket handler."""
    global training_handler
    if training_handler is None:
        raise RuntimeError("Training handler not initialized")
    return training_handler


def initialize_training_handler(training_service: AutomatedTrainingService):
    """Initialize the global training WebSocket handler.

    Args:
        training_service: Automated training service
    """
    global training_handler
    training_handler = TrainingWebSocketHandler(training_service)


# WebSocket endpoint function
async def training_websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for training operations.

    Args:
        websocket: WebSocket connection
        client_id: Unique client identifier
    """
    handler = get_training_handler()

    try:
        await handler.connect(websocket, client_id)

        while True:
            # Receive message
            data = await websocket.receive_text()

            try:
                message = json.loads(data)
                await handler.handle_message(websocket, client_id, message)
            except json.JSONDecodeError:
                await handler._send_error(websocket, "Invalid JSON format")

    except WebSocketDisconnect:
        await handler.disconnect(client_id)
    except Exception as e:
        logger.error(f"Training WebSocket error: {str(e)}")
        await handler.disconnect(client_id)
