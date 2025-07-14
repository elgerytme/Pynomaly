"""Task processing service with worker management."""

from __future__ import annotations

import asyncio
import logging
import signal
import time
from typing import Any, Callable

from ..config.messaging_settings import MessagingSettings
from ..models.messages import Message
from ..models.tasks import Task, TaskStatus
from ..protocols.message_queue_protocol import MessageQueueProtocol

logger = logging.getLogger(__name__)


class TaskProcessor:
    """Task processing service with worker management."""

    def __init__(self, adapter: MessageQueueProtocol, settings: MessagingSettings):
        """Initialize task processor.

        Args:
            adapter: Message queue adapter implementation
            settings: Messaging settings
        """
        self._adapter = adapter
        self._settings = settings
        self._workers: list[asyncio.Task] = []
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._task_registry: dict[str, Callable] = {}
        self._stats = {
            "tasks_processed": 0,
            "tasks_failed": 0,
            "tasks_retried": 0,
            "processing_time_total": 0.0,
            "started_at": None,
        }

    def register_task_handler(self, task_type: str, handler: Callable) -> None:
        """Register a task handler function.

        Args:
            task_type: Type of task to handle
            handler: Async function to handle the task
        """
        self._task_registry[task_type] = handler
        logger.info(f"Registered handler for task type: {task_type}")

    async def start(self, queue_names: list[str] | None = None) -> None:
        """Start the task processor.

        Args:
            queue_names: List of queue names to process (default: ["default"])
        """
        if self._running:
            logger.warning("Task processor is already running")
            return

        if queue_names is None:
            queue_names = ["default"]

        try:
            await self._adapter.connect()
            self._running = True
            self._stats["started_at"] = time.time()

            # Start worker tasks
            for i in range(self._settings.worker_concurrency):
                for queue_name in queue_names:
                    worker = asyncio.create_task(
                        self._worker_loop(f"worker-{i}-{queue_name}", queue_name)
                    )
                    self._workers.append(worker)

            # Start monitoring task
            monitor_task = asyncio.create_task(self._monitor_loop())
            self._workers.append(monitor_task)

            # Setup signal handlers
            self._setup_signal_handlers()

            logger.info(
                f"Task processor started with {self._settings.worker_concurrency} workers "
                f"for queues: {queue_names}"
            )

        except Exception as e:
            logger.error(f"Failed to start task processor: {e}")
            await self.stop()
            raise

    async def stop(self) -> None:
        """Stop the task processor."""
        if not self._running:
            return

        logger.info("Stopping task processor...")
        self._running = False
        self._shutdown_event.set()

        # Cancel all workers
        for worker in self._workers:
            worker.cancel()

        # Wait for workers to finish with timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(*self._workers, return_exceptions=True),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            logger.warning("Some workers did not shut down gracefully")

        # Disconnect from queue
        try:
            await self._adapter.disconnect()
        except Exception as e:
            logger.error(f"Error disconnecting from queue: {e}")

        self._workers.clear()
        logger.info("Task processor stopped")

    async def _worker_loop(self, worker_name: str, queue_name: str) -> None:
        """Main worker processing loop.

        Args:
            worker_name: Name of the worker
            queue_name: Name of the queue to process
        """
        logger.info(f"Worker {worker_name} started for queue {queue_name}")

        while self._running and not self._shutdown_event.is_set():
            try:
                # Receive message from queue
                message = await self._adapter.receive_message(
                    queue_name, timeout=self._settings.redis_block_timeout // 1000
                )

                if message is None:
                    continue

                # Process the message
                await self._process_message(worker_name, message)

            except asyncio.CancelledError:
                logger.info(f"Worker {worker_name} cancelled")
                break
            except Exception as e:
                logger.error(f"Error in worker {worker_name}: {e}")
                await asyncio.sleep(1)  # Brief pause before retrying

        logger.info(f"Worker {worker_name} stopped")

    async def _process_message(self, worker_name: str, message: Message) -> None:
        """Process a single message.

        Args:
            worker_name: Name of the worker processing the message
            message: Message to process
        """
        start_time = time.time()
        task_id = message.payload.get("task_id")

        try:
            logger.debug(f"Worker {worker_name} processing message {message.id}")

            # Get task details if this is a task message
            if message.message_type == "task" and task_id:
                task = await self._adapter.get_task_status(task_id)
                if task:
                    await self._process_task(worker_name, task, message)
                else:
                    logger.error(f"Task {task_id} not found")
                    await self._adapter.reject_message(message, requeue=False)
            else:
                # Handle non-task messages
                await self._process_generic_message(worker_name, message)

            # Acknowledge successful processing
            await self._adapter.acknowledge_message(message)
            
            # Update statistics
            processing_time = time.time() - start_time
            self._stats["tasks_processed"] += 1
            self._stats["processing_time_total"] += processing_time

            logger.debug(
                f"Worker {worker_name} completed message {message.id} "
                f"in {processing_time:.2f}s"
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(
                f"Worker {worker_name} failed to process message {message.id}: {e}"
            )

            # Update task status if applicable
            if task_id:
                task = await self._adapter.get_task_status(task_id)
                if task:
                    task.mark_failed(str(e))
                    # Update task in storage would go here

            # Handle message retry logic
            if message.can_retry():
                self._stats["tasks_retried"] += 1
                await self._adapter.reject_message(message, requeue=True)
            else:
                self._stats["tasks_failed"] += 1
                await self._adapter.reject_message(message, requeue=False)

    async def _process_task(self, worker_name: str, task: Task, message: Message) -> None:
        """Process a task message.

        Args:
            worker_name: Name of the worker
            task: Task to process
            message: Original message
        """
        # Check if we have a handler for this task type
        handler = self._task_registry.get(task.task_type.value)
        if not handler:
            raise ValueError(f"No handler registered for task type: {task.task_type}")

        # Update task status
        task.mark_running()
        # In a real implementation, you'd update this in persistent storage

        try:
            # Execute the task with timeout
            if task.timeout:
                result = await asyncio.wait_for(
                    handler(*task.args, **task.kwargs),
                    timeout=task.timeout
                )
            else:
                result = await handler(*task.args, **task.kwargs)

            # Mark task as completed
            task.mark_completed(result)
            logger.info(f"Task {task.id} completed successfully by {worker_name}")

        except asyncio.TimeoutError:
            task.mark_timeout()
            logger.error(f"Task {task.id} timed out after {task.timeout}s")
            raise
        except Exception as e:
            task.mark_failed(str(e))
            logger.error(f"Task {task.id} failed: {e}")
            raise

    async def _process_generic_message(self, worker_name: str, message: Message) -> None:
        """Process a generic (non-task) message.

        Args:
            worker_name: Name of the worker
            message: Message to process
        """
        # Handle generic messages based on message type or content
        message_type = message.message_type or "generic"
        
        # You can extend this to handle different message types
        logger.info(
            f"Worker {worker_name} processed generic message of type: {message_type}"
        )

    async def _monitor_loop(self) -> None:
        """Monitor worker health and performance."""
        logger.info("Task processor monitor started")

        while self._running and not self._shutdown_event.is_set():
            try:
                # Log statistics periodically
                if self._settings.metrics_interval > 0:
                    await asyncio.sleep(self._settings.metrics_interval)
                    await self._log_statistics()
                else:
                    await asyncio.sleep(60)  # Default 1 minute

                # Check worker health
                active_workers = sum(1 for w in self._workers if not w.done())
                if active_workers < len(self._workers) - 1:  # -1 for monitor task
                    logger.warning(f"Only {active_workers} workers active")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                await asyncio.sleep(5)

        logger.info("Task processor monitor stopped")

    async def _log_statistics(self) -> None:
        """Log processing statistics."""
        if self._stats["started_at"]:
            uptime = time.time() - self._stats["started_at"]
            avg_processing_time = (
                self._stats["processing_time_total"] / max(1, self._stats["tasks_processed"])
            )
            
            logger.info(
                f"Task processor stats - "
                f"Uptime: {uptime:.0f}s, "
                f"Processed: {self._stats['tasks_processed']}, "
                f"Failed: {self._stats['tasks_failed']}, "
                f"Retried: {self._stats['tasks_retried']}, "
                f"Avg time: {avg_processing_time:.2f}s"
            )

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(sig, frame):
            logger.info(f"Received signal {sig}, initiating shutdown...")
            asyncio.create_task(self.stop())

        try:
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
        except ValueError:
            # Signal handling not available (e.g., running in thread)
            logger.warning("Signal handling not available")

    def get_statistics(self) -> dict[str, Any]:
        """Get current processing statistics.

        Returns:
            Dictionary with current statistics
        """
        stats = self._stats.copy()
        if stats["started_at"]:
            stats["uptime"] = time.time() - stats["started_at"]
            stats["avg_processing_time"] = (
                stats["processing_time_total"] / max(1, stats["tasks_processed"])
            )
        stats["workers_running"] = len([w for w in self._workers if not w.done()])
        stats["is_running"] = self._running
        return stats