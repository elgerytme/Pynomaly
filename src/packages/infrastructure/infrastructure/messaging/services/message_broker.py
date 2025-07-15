"""Message broker service for routing and handling messages."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

from ..config.messaging_settings import MessagingSettings
from ..models.messages import Message, MessagePriority
from ..models.tasks import Task, TaskType
from ..protocols.message_queue_protocol import MessageQueueProtocol

logger = logging.getLogger(__name__)


class MessageBroker:
    """Message broker service for routing and handling messages."""

    def __init__(self, adapter: MessageQueueProtocol, settings: MessagingSettings):
        """Initialize message broker.

        Args:
            adapter: Message queue adapter implementation
            settings: Messaging settings
        """
        self._adapter = adapter
        self._settings = settings
        self._message_handlers: dict[str, Callable] = {}
        self._routing_rules: dict[str, str] = {}  # message_type -> queue_name
        self._connected = False

    async def start(self) -> None:
        """Start the message broker."""
        try:
            await self._adapter.connect()
            self._connected = True
            
            # Set up default routing rules
            self._setup_default_routing()
            
            logger.info("Message broker started successfully")
        except Exception as e:
            logger.error(f"Failed to start message broker: {e}")
            raise

    async def stop(self) -> None:
        """Stop the message broker."""
        try:
            await self._adapter.disconnect()
            self._connected = False
            logger.info("Message broker stopped")
        except Exception as e:
            logger.error(f"Error stopping message broker: {e}")

    def _setup_default_routing(self) -> None:
        """Set up default message routing rules."""
        self._routing_rules.update({
            "task": "tasks",
            "notification": "notifications",
            "event": "events",
            "anomaly_detection": "anomaly_detection",
            "data_profiling": "data_profiling",
            "model_training": "model_training",
            "report_generation": "reports",
            "cleanup": "maintenance",
        })

    def register_message_handler(self, message_type: str, handler: Callable) -> None:
        """Register a message handler.

        Args:
            message_type: Type of message to handle
            handler: Function to handle the message
        """
        self._message_handlers[message_type] = handler
        logger.info(f"Registered handler for message type: {message_type}")

    def add_routing_rule(self, message_type: str, queue_name: str) -> None:
        """Add a message routing rule.

        Args:
            message_type: Type of message
            queue_name: Queue to route to
        """
        self._routing_rules[message_type] = queue_name
        logger.info(f"Added routing rule: {message_type} -> {queue_name}")

    async def publish_message(
        self,
        message_type: str,
        payload: dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
        **kwargs
    ) -> str:
        """Publish a message to the appropriate queue.

        Args:
            message_type: Type of message
            payload: Message payload
            priority: Message priority
            **kwargs: Additional message options

        Returns:
            Message ID
        """
        if not self._connected:
            raise RuntimeError("Message broker not connected")

        # Determine target queue
        queue_name = self._routing_rules.get(message_type, "default")

        # Create message
        message = Message(
            queue_name=queue_name,
            payload=payload,
            priority=priority,
            message_type=message_type,
            **kwargs
        )

        # Send message
        success = await self._adapter.send_message(queue_name, message)
        if not success:
            raise RuntimeError(f"Failed to publish message of type {message_type}")

        logger.debug(f"Published {message_type} message {message.id} to {queue_name}")
        return message.id

    async def publish_task(self, task: Task) -> str:
        """Publish a task for processing.

        Args:
            task: Task to publish

        Returns:
            Task ID
        """
        if not self._connected:
            raise RuntimeError("Message broker not connected")

        # Route task to appropriate queue
        queue_name = self._get_task_queue(task.task_type)
        task.queue_name = queue_name

        task_id = await self._adapter.submit_task(task)
        logger.info(f"Published task {task_id} of type {task.task_type} to {queue_name}")
        return task_id

    def _get_task_queue(self, task_type: TaskType) -> str:
        """Get the appropriate queue for a task type.

        Args:
            task_type: Type of task

        Returns:
            Queue name
        """
        # Map task types to queues
        task_queue_mapping = {
            TaskType.ANOMALY_DETECTION: "anomaly_detection",
            TaskType.DATA_PROFILING: "data_profiling", 
            TaskType.MODEL_TRAINING: "model_training",
            TaskType.DATA_PROCESSING: "data_processing",
            TaskType.REPORT_GENERATION: "reports",
            TaskType.NOTIFICATION: "notifications",
            TaskType.CLEANUP: "maintenance",
            TaskType.EXPORT: "exports",
            TaskType.IMPORT: "imports",
            TaskType.VALIDATION: "validation",
        }
        
        return task_queue_mapping.get(task_type, "default")

    async def publish_notification(
        self,
        recipient: str,
        subject: str,
        content: str,
        notification_type: str = "email",
        priority: MessagePriority = MessagePriority.NORMAL
    ) -> str:
        """Publish a notification message.

        Args:
            recipient: Notification recipient
            subject: Notification subject
            content: Notification content
            notification_type: Type of notification (email, sms, push)
            priority: Message priority

        Returns:
            Message ID
        """
        payload = {
            "recipient": recipient,
            "subject": subject,
            "content": content,
            "notification_type": notification_type,
        }

        return await self.publish_message(
            message_type="notification",
            payload=payload,
            priority=priority
        )

    async def publish_event(
        self,
        event_type: str,
        data: dict[str, Any],
        source: str | None = None,
        priority: MessagePriority = MessagePriority.NORMAL
    ) -> str:
        """Publish an event message.

        Args:
            event_type: Type of event
            data: Event data
            source: Event source
            priority: Message priority

        Returns:
            Message ID
        """
        payload = {
            "event_type": event_type,
            "data": data,
            "source": source,
        }

        return await self.publish_message(
            message_type="event",
            payload=payload,
            priority=priority
        )

    async def schedule_task(
        self,
        task: Task,
        delay_seconds: int | None = None,
        schedule_at: str | None = None
    ) -> str:
        """Schedule a task for future execution.

        Args:
            task: Task to schedule
            delay_seconds: Delay in seconds from now
            schedule_at: ISO timestamp to schedule at

        Returns:
            Task ID
        """
        if delay_seconds is not None:
            from datetime import datetime, timezone, timedelta
            task.scheduled_at = datetime.now(timezone.utc) + timedelta(seconds=delay_seconds)
        elif schedule_at is not None:
            from datetime import datetime
            task.scheduled_at = datetime.fromisoformat(schedule_at)

        return await self.publish_task(task)

    async def create_anomaly_detection_task(
        self,
        dataset_id: str,
        algorithm: str,
        parameters: dict[str, Any] | None = None,
        priority: int = 0
    ) -> str:
        """Create an anomaly detection task.

        Args:
            dataset_id: ID of the dataset to analyze
            algorithm: Anomaly detection algorithm to use
            parameters: Algorithm parameters
            priority: Task priority

        Returns:
            Task ID
        """
        task = Task(
            task_type=TaskType.ANOMALY_DETECTION,
            name=f"Anomaly Detection - {dataset_id}",
            description=f"Run {algorithm} on dataset {dataset_id}",
            function_name="detect_anomalies",
            kwargs={
                "dataset_id": dataset_id,
                "algorithm": algorithm,
                "parameters": parameters or {},
            },
            priority=priority,
        )

        return await self.publish_task(task)

    async def create_data_profiling_task(
        self,
        dataset_id: str,
        profile_types: list[str] | None = None,
        priority: int = 0
    ) -> str:
        """Create a data profiling task.

        Args:
            dataset_id: ID of the dataset to profile
            profile_types: Types of profiling to perform
            priority: Task priority

        Returns:
            Task ID
        """
        task = Task(
            task_type=TaskType.DATA_PROFILING,
            name=f"Data Profiling - {dataset_id}",
            description=f"Profile dataset {dataset_id}",
            function_name="profile_dataset",
            kwargs={
                "dataset_id": dataset_id,
                "profile_types": profile_types or ["basic", "advanced"],
            },
            priority=priority,
        )

        return await self.publish_task(task)

    async def create_model_training_task(
        self,
        model_id: str,
        training_data_id: str,
        algorithm: str,
        hyperparameters: dict[str, Any] | None = None,
        priority: int = 0
    ) -> str:
        """Create a model training task.

        Args:
            model_id: ID of the model to train
            training_data_id: ID of the training dataset
            algorithm: Training algorithm
            hyperparameters: Algorithm hyperparameters
            priority: Task priority

        Returns:
            Task ID
        """
        task = Task(
            task_type=TaskType.MODEL_TRAINING,
            name=f"Model Training - {model_id}",
            description=f"Train model {model_id} with {algorithm}",
            function_name="train_model",
            kwargs={
                "model_id": model_id,
                "training_data_id": training_data_id,
                "algorithm": algorithm,
                "hyperparameters": hyperparameters or {},
            },
            priority=priority,
            timeout=3600,  # 1 hour timeout for training
        )

        return await self.publish_task(task)

    async def create_report_generation_task(
        self,
        report_type: str,
        data_source: str,
        parameters: dict[str, Any] | None = None,
        format: str = "pdf",
        priority: int = 0
    ) -> str:
        """Create a report generation task.

        Args:
            report_type: Type of report to generate
            data_source: Data source for the report
            parameters: Report parameters
            format: Report format (pdf, html, csv)
            priority: Task priority

        Returns:
            Task ID
        """
        task = Task(
            task_type=TaskType.REPORT_GENERATION,
            name=f"Report Generation - {report_type}",
            description=f"Generate {report_type} report from {data_source}",
            function_name="generate_report",
            kwargs={
                "report_type": report_type,
                "data_source": data_source,
                "parameters": parameters or {},
                "format": format,
            },
            priority=priority,
        )

        return await self.publish_task(task)

    async def get_queue_statistics(self) -> dict[str, Any]:
        """Get statistics for all queues.

        Returns:
            Dictionary with statistics for each queue
        """
        if not self._connected:
            return {}

        # Get statistics for all known queues
        queue_names = set(self._routing_rules.values())
        queue_names.add("default")

        stats = {}
        for queue_name in queue_names:
            try:
                queue_stats = await self._adapter.get_queue_stats(queue_name)
                stats[queue_name] = queue_stats
            except Exception as e:
                logger.error(f"Failed to get stats for queue {queue_name}: {e}")
                stats[queue_name] = {"error": str(e)}

        return stats

    async def health_check(self) -> dict[str, Any]:
        """Perform a health check of the message broker.

        Returns:
            Health check results
        """
        health = {
            "connected": self._connected,
            "adapter_healthy": False,
            "registered_handlers": len(self._message_handlers),
            "routing_rules": len(self._routing_rules),
        }

        if self._connected:
            try:
                health["adapter_healthy"] = await self._adapter.health_check()
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                health["error"] = str(e)

        return health