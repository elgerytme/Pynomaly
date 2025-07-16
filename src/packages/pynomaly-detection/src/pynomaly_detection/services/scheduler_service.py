"""
Scheduler service for automated reporting and export operations.

This service provides:
- Scheduled report generation
- Automated export workflows
- Cron-based scheduling
- Email notifications
- Report distribution
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from croniter import croniter

from ...shared.exceptions import SchedulerError, ValidationError
from ..dto.export_options import ExportOptions
from .export_service import ExportService
from .template_service import TemplateService


@dataclass
class ScheduleConfig:
    """Configuration for scheduled tasks."""

    schedule_id: str
    name: str
    description: str

    # Schedule configuration
    cron_expression: str
    timezone: str = "UTC"
    enabled: bool = True

    # Task configuration
    task_type: str = "export"  # export, report, notification
    template_id: str | None = None
    export_format: str = "excel"

    # Data source configuration
    data_source: dict[str, Any] = None
    data_filters: dict[str, Any] = None

    # Export configuration
    export_options: dict[str, Any] = None
    destination: dict[str, Any] = None

    # Notification configuration
    notify_on_success: bool = True
    notify_on_failure: bool = True
    notification_recipients: list[str] = None

    # Metadata
    created_at: datetime | None = None
    updated_at: datetime | None = None
    last_run: datetime | None = None
    next_run: datetime | None = None
    run_count: int = 0
    success_count: int = 0
    failure_count: int = 0

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = self.created_at
        if self.data_source is None:
            self.data_source = {}
        if self.data_filters is None:
            self.data_filters = {}
        if self.export_options is None:
            self.export_options = {}
        if self.destination is None:
            self.destination = {}
        if self.notification_recipients is None:
            self.notification_recipients = []

        # Calculate next run time
        if self.next_run is None:
            self._calculate_next_run()

    def _calculate_next_run(self):
        """Calculate next run time based on cron expression."""
        try:
            cron = croniter(self.cron_expression, datetime.now())
            self.next_run = cron.get_next(datetime)
        except Exception:
            self.next_run = None


@dataclass
class ScheduleExecutionResult:
    """Result of a scheduled task execution."""

    schedule_id: str
    execution_id: str
    start_time: datetime
    end_time: datetime | None = None
    status: str = "running"  # running, success, failed
    result: dict[str, Any] | None = None
    error: str | None = None
    logs: list[str] = None

    def __post_init__(self):
        if self.logs is None:
            self.logs = []


class SchedulerService:
    """Service for managing scheduled tasks and automated reporting."""

    def __init__(
        self,
        export_service: ExportService,
        template_service: TemplateService,
        notification_service: Any | None = None,
    ):
        """Initialize scheduler service."""
        self.logger = logging.getLogger(__name__)
        self.export_service = export_service
        self.template_service = template_service
        self.notification_service = notification_service

        # Schedule registry
        self._schedules: dict[str, ScheduleConfig] = {}
        self._running_tasks: dict[str, asyncio.Task] = {}
        self._execution_history: list[ScheduleExecutionResult] = []

        # Scheduler state
        self._scheduler_running = False
        self._scheduler_task: asyncio.Task | None = None

    def create_schedule(
        self, name: str, cron_expression: str, task_type: str = "export", **kwargs
    ) -> ScheduleConfig:
        """Create a new scheduled task."""
        try:
            # Validate cron expression
            if not croniter.is_valid(cron_expression):
                raise ValidationError(f"Invalid cron expression: {cron_expression}")

            schedule_id = kwargs.get("schedule_id", str(uuid.uuid4()))

            schedule_config = ScheduleConfig(
                schedule_id=schedule_id,
                name=name,
                cron_expression=cron_expression,
                task_type=task_type,
                **kwargs,
            )

            self._schedules[schedule_id] = schedule_config

            self.logger.info(f"Created schedule: {name} ({schedule_id})")
            return schedule_config

        except Exception as e:
            raise SchedulerError(f"Failed to create schedule: {str(e)}") from e

    def update_schedule(self, schedule_id: str, **updates) -> ScheduleConfig:
        """Update an existing schedule."""
        try:
            schedule = self._schedules.get(schedule_id)
            if not schedule:
                raise ValidationError(f"Schedule not found: {schedule_id}")

            # Validate cron expression if being updated
            if "cron_expression" in updates:
                if not croniter.is_valid(updates["cron_expression"]):
                    raise ValidationError(
                        f"Invalid cron expression: {updates['cron_expression']}"
                    )

            # Update fields
            for key, value in updates.items():
                if hasattr(schedule, key):
                    setattr(schedule, key, value)

            schedule.updated_at = datetime.now()
            schedule._calculate_next_run()

            self.logger.info(f"Updated schedule: {schedule.name} ({schedule_id})")
            return schedule

        except Exception as e:
            raise SchedulerError(f"Failed to update schedule: {str(e)}") from e

    def delete_schedule(self, schedule_id: str) -> bool:
        """Delete a scheduled task."""
        try:
            if schedule_id not in self._schedules:
                return False

            # Stop running task if any
            if schedule_id in self._running_tasks:
                self._running_tasks[schedule_id].cancel()
                del self._running_tasks[schedule_id]

            # Remove schedule
            del self._schedules[schedule_id]

            self.logger.info(f"Deleted schedule: {schedule_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to delete schedule {schedule_id}: {str(e)}")
            return False

    def get_schedule(self, schedule_id: str) -> ScheduleConfig | None:
        """Get schedule by ID."""
        return self._schedules.get(schedule_id)

    def list_schedules(
        self, enabled_only: bool = False, task_type: str | None = None
    ) -> list[ScheduleConfig]:
        """List all schedules with optional filtering."""
        schedules = list(self._schedules.values())

        if enabled_only:
            schedules = [s for s in schedules if s.enabled]

        if task_type:
            schedules = [s for s in schedules if s.task_type == task_type]

        return sorted(schedules, key=lambda s: s.next_run or datetime.max)

    async def start_scheduler(self) -> None:
        """Start the scheduler."""
        if self._scheduler_running:
            self.logger.warning("Scheduler is already running")
            return

        self._scheduler_running = True
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        self.logger.info("Scheduler started")

    async def stop_scheduler(self) -> None:
        """Stop the scheduler."""
        if not self._scheduler_running:
            return

        self._scheduler_running = False

        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass

        # Cancel all running tasks
        for task in self._running_tasks.values():
            task.cancel()

        if self._running_tasks:
            await asyncio.gather(*self._running_tasks.values(), return_exceptions=True)

        self._running_tasks.clear()
        self.logger.info("Scheduler stopped")

    async def execute_schedule_now(self, schedule_id: str) -> ScheduleExecutionResult:
        """Execute a schedule immediately."""
        schedule = self._schedules.get(schedule_id)
        if not schedule:
            raise ValidationError(f"Schedule not found: {schedule_id}")

        return await self._execute_schedule(schedule)

    def get_execution_history(
        self, schedule_id: str | None = None, limit: int = 100
    ) -> list[ScheduleExecutionResult]:
        """Get execution history."""
        history = self._execution_history

        if schedule_id:
            history = [h for h in history if h.schedule_id == schedule_id]

        return sorted(history, key=lambda h: h.start_time, reverse=True)[:limit]

    async def _scheduler_loop(self) -> None:
        """Main scheduler loop."""
        while self._scheduler_running:
            try:
                await self._check_and_execute_schedules()
                await asyncio.sleep(60)  # Check every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Scheduler loop error: {str(e)}")
                await asyncio.sleep(60)

    async def _check_and_execute_schedules(self) -> None:
        """Check for schedules that need to be executed."""
        now = datetime.now()

        for schedule in self._schedules.values():
            if not schedule.enabled:
                continue

            if schedule.next_run and now >= schedule.next_run:
                if schedule.schedule_id not in self._running_tasks:
                    task = asyncio.create_task(self._execute_schedule(schedule))
                    self._running_tasks[schedule.schedule_id] = task

                # Calculate next run time
                schedule._calculate_next_run()

    async def _execute_schedule(
        self, schedule: ScheduleConfig
    ) -> ScheduleExecutionResult:
        """Execute a scheduled task."""
        execution_id = str(uuid.uuid4())
        execution_result = ScheduleExecutionResult(
            schedule_id=schedule.schedule_id,
            execution_id=execution_id,
            start_time=datetime.now(),
        )

        try:
            self.logger.info(
                f"Executing schedule: {schedule.name} ({schedule.schedule_id})"
            )
            execution_result.logs.append(f"Started execution of {schedule.name}")

            # Execute based on task type
            if schedule.task_type == "export":
                result = await self._execute_export_task(schedule, execution_result)
            elif schedule.task_type == "report":
                result = await self._execute_report_task(schedule, execution_result)
            elif schedule.task_type == "notification":
                result = await self._execute_notification_task(
                    schedule, execution_result
                )
            else:
                raise ValidationError(f"Unknown task type: {schedule.task_type}")

            execution_result.result = result
            execution_result.status = "success"
            execution_result.end_time = datetime.now()

            # Update schedule stats
            schedule.last_run = execution_result.start_time
            schedule.run_count += 1
            schedule.success_count += 1

            # Send success notification
            if schedule.notify_on_success:
                await self._send_notification(schedule, execution_result, "success")

            execution_result.logs.append("Execution completed successfully")
            self.logger.info(f"Schedule executed successfully: {schedule.name}")

        except Exception as e:
            execution_result.status = "failed"
            execution_result.error = str(e)
            execution_result.end_time = datetime.now()

            # Update schedule stats
            schedule.last_run = execution_result.start_time
            schedule.run_count += 1
            schedule.failure_count += 1

            # Send failure notification
            if schedule.notify_on_failure:
                await self._send_notification(schedule, execution_result, "failure")

            execution_result.logs.append(f"Execution failed: {str(e)}")
            self.logger.error(f"Schedule execution failed: {schedule.name} - {str(e)}")

        finally:
            # Clean up running task
            if schedule.schedule_id in self._running_tasks:
                del self._running_tasks[schedule.schedule_id]

            # Store execution result
            self._execution_history.append(execution_result)

            # Keep only last 1000 executions
            if len(self._execution_history) > 1000:
                self._execution_history = self._execution_history[-1000:]

        return execution_result

    async def _execute_export_task(
        self, schedule: ScheduleConfig, execution_result: ScheduleExecutionResult
    ) -> dict[str, Any]:
        """Execute an export task."""
        execution_result.logs.append("Starting export task")

        # Get data source
        data = await self._get_data_from_source(
            schedule.data_source, schedule.data_filters
        )
        execution_result.logs.append(f"Retrieved {len(data)} records from data source")

        # Create export options
        export_options = ExportOptions(**schedule.export_options)

        # Execute export
        result = await self.export_service.export_results(
            data=data,
            format_type=schedule.export_format,
            options=export_options,
            **schedule.destination,
        )

        execution_result.logs.append(
            f"Export completed: {result.get('file_path', 'N/A')}"
        )
        return result

    async def _execute_report_task(
        self, schedule: ScheduleConfig, execution_result: ScheduleExecutionResult
    ) -> dict[str, Any]:
        """Execute a report generation task."""
        execution_result.logs.append("Starting report generation task")

        if not schedule.template_id:
            raise ValidationError("Template ID required for report tasks")

        # Get data source
        data = await self._get_data_from_source(
            schedule.data_source, schedule.data_filters
        )
        execution_result.logs.append(f"Retrieved {len(data)} records from data source")

        # Render template
        result = self.template_service.render_template(
            template_id=schedule.template_id,
            data=data,
            context={"schedule_name": schedule.name, "execution_time": datetime.now()},
        )

        # Export rendered report
        if schedule.export_format and schedule.destination:
            export_options = ExportOptions(**schedule.export_options)
            export_result = await self.export_service.export_results(
                data=data,
                format_type=schedule.export_format,
                options=export_options,
                template_result=result,
                **schedule.destination,
            )
            result.update(export_result)

        execution_result.logs.append("Report generation completed")
        return result

    async def _execute_notification_task(
        self, schedule: ScheduleConfig, execution_result: ScheduleExecutionResult
    ) -> dict[str, Any]:
        """Execute a notification task."""
        execution_result.logs.append("Starting notification task")

        # Get data for notification
        data = await self._get_data_from_source(
            schedule.data_source, schedule.data_filters
        )

        # Send notification
        if self.notification_service:
            result = await self.notification_service.send_notification(
                recipients=schedule.notification_recipients,
                subject=f"Scheduled notification: {schedule.name}",
                data=data,
                **schedule.destination,
            )
        else:
            result = {"status": "notification_service_not_configured"}

        execution_result.logs.append("Notification sent")
        return result

    async def _get_data_from_source(
        self, data_source: dict[str, Any], filters: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Get data from configured data source."""
        source_type = data_source.get("type", "mock")

        if source_type == "mock":
            # Generate mock data for testing
            import random

            data = []
            for i in range(100):
                data.append(
                    {
                        "timestamp": datetime.now() - timedelta(hours=i),
                        "anomaly_score": random.random(),
                        "is_anomaly": random.random() > 0.8,
                        "feature_1": random.normalvariate(0, 1),
                        "feature_2": random.normalvariate(0, 1),
                    }
                )
            return data

        elif source_type == "database":
            # TODO: Implement database data source
            raise NotImplementedError("Database data source not implemented")

        elif source_type == "api":
            # TODO: Implement API data source
            raise NotImplementedError("API data source not implemented")

        elif source_type == "file":
            # TODO: Implement file data source
            raise NotImplementedError("File data source not implemented")

        else:
            raise ValidationError(f"Unknown data source type: {source_type}")

    async def _send_notification(
        self,
        schedule: ScheduleConfig,
        execution_result: ScheduleExecutionResult,
        notification_type: str,
    ) -> None:
        """Send notification about schedule execution."""
        if not self.notification_service or not schedule.notification_recipients:
            return

        try:
            subject = f"Schedule {notification_type}: {schedule.name}"

            if notification_type == "success":
                message = (
                    f"Schedule '{schedule.name}' executed successfully at "
                    f"{execution_result.start_time}"
                )
            else:
                message = (
                    f"Schedule '{schedule.name}' failed at "
                    f"{execution_result.start_time}. Error: {execution_result.error}"
                )

            await self.notification_service.send_notification(
                recipients=schedule.notification_recipients,
                subject=subject,
                message=message,
                execution_result=execution_result,
            )

        except Exception as e:
            self.logger.error(f"Failed to send notification: {str(e)}")

    def create_common_schedules(self) -> list[ScheduleConfig]:
        """Create common schedule templates."""
        common_schedules = [
            {
                "name": "Daily Anomaly Report",
                "description": "Daily summary of anomaly detection results",
                "cron_expression": "0 9 * * *",  # Daily at 9 AM
                "task_type": "report",
                "export_format": "excel",
                "destination": {"type": "email", "recipients": ["admin@company.com"]},
            },
            {
                "name": "Weekly Executive Summary",
                "description": "Weekly executive summary report",
                "cron_expression": "0 10 * * 1",  # Monday at 10 AM
                "task_type": "report",
                "export_format": "pdf",
                "destination": {
                    "type": "email",
                    "recipients": ["executives@company.com"],
                },
            },
            {
                "name": "Hourly High Priority Alerts",
                "description": "Hourly check for high priority anomalies",
                "cron_expression": "0 * * * *",  # Every hour
                "task_type": "notification",
                "data_filters": {
                    "anomaly_score": {"type": "greater_than", "value": 0.8}
                },
                "destination": {"type": "slack", "channel": "#alerts"},
            },
        ]

        created_schedules = []
        for schedule_data in common_schedules:
            try:
                schedule = self.create_schedule(**schedule_data)
                created_schedules.append(schedule)
            except Exception as e:
                self.logger.error(
                    f"Failed to create common schedule {schedule_data['name']}: "
                    f"{str(e)}"
                )

        return created_schedules
