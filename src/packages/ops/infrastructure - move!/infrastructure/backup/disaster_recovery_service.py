#!/usr/bin/env python3
"""
Disaster Recovery Service for Pynomaly

This module provides comprehensive disaster recovery capabilities including
automated recovery plans, failover management, and business continuity features.
"""

import asyncio
import logging
import os
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from .backup_manager import BackupManager, BackupStatus, BackupType


class RecoveryPriority(Enum):
    """Recovery priority levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RecoveryStatus(Enum):
    """Recovery operation status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PARTIAL = "partial"


class FailoverStatus(Enum):
    """Failover status."""

    ACTIVE = "active"
    STANDBY = "standby"
    FAILED = "failed"
    MAINTENANCE = "maintenance"


@dataclass
class RecoveryPoint:
    """Recovery point objective configuration."""

    service_name: str
    rpo_minutes: int  # Recovery Point Objective in minutes
    rto_minutes: int  # Recovery Time Objective in minutes
    priority: RecoveryPriority
    backup_frequency_minutes: int
    dependencies: list[str] = field(default_factory=list)
    health_check_url: str | None = None
    restart_command: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "service_name": self.service_name,
            "rpo_minutes": self.rpo_minutes,
            "rto_minutes": self.rto_minutes,
            "priority": self.priority.value,
            "backup_frequency_minutes": self.backup_frequency_minutes,
            "dependencies": self.dependencies,
            "health_check_url": self.health_check_url,
            "restart_command": self.restart_command,
        }


@dataclass
class RecoveryPlan:
    """Disaster recovery plan."""

    plan_id: str
    name: str
    description: str
    recovery_points: list[RecoveryPoint]
    notification_channels: list[str] = field(default_factory=list)
    pre_recovery_scripts: list[str] = field(default_factory=list)
    post_recovery_scripts: list[str] = field(default_factory=list)
    estimated_rto_minutes: int = 0
    last_tested: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "plan_id": self.plan_id,
            "name": self.name,
            "description": self.description,
            "recovery_points": [rp.to_dict() for rp in self.recovery_points],
            "notification_channels": self.notification_channels,
            "pre_recovery_scripts": self.pre_recovery_scripts,
            "post_recovery_scripts": self.post_recovery_scripts,
            "estimated_rto_minutes": self.estimated_rto_minutes,
            "last_tested": self.last_tested.isoformat() if self.last_tested else None,
        }


@dataclass
class RecoveryOperation:
    """Recovery operation tracking."""

    operation_id: str
    plan_id: str
    status: RecoveryStatus
    start_time: datetime
    end_time: datetime | None = None
    triggered_by: str = "manual"
    error_message: str = ""
    recovered_services: list[str] = field(default_factory=list)
    failed_services: list[str] = field(default_factory=list)

    @property
    def duration_minutes(self) -> float:
        """Get operation duration in minutes."""
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds() / 60

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "operation_id": self.operation_id,
            "plan_id": self.plan_id,
            "status": self.status.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "triggered_by": self.triggered_by,
            "error_message": self.error_message,
            "recovered_services": self.recovered_services,
            "failed_services": self.failed_services,
            "duration_minutes": self.duration_minutes,
        }


class NotificationChannel(ABC):
    """Base class for notification channels."""

    def __init__(self, name: str, config: dict[str, Any]):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    async def send_notification(
        self, title: str, message: str, severity: str = "info"
    ) -> bool:
        """Send notification."""
        pass


class EmailNotificationChannel(NotificationChannel):
    """Email notification channel."""

    async def send_notification(
        self, title: str, message: str, severity: str = "info"
    ) -> bool:
        """Send email notification."""
        try:
            import smtplib
            from email.mime.multipart import MIMEMultipart
            from email.mime.text import MIMEText

            smtp_server = self.config.get("smtp_server")
            smtp_port = self.config.get("smtp_port", 587)
            username = self.config.get("username")
            password = self.config.get("password")
            from_email = self.config.get("from_email")
            recipients = self.config.get("recipients", [])

            if not all([smtp_server, username, password, from_email, recipients]):
                self.logger.warning("Email configuration incomplete")
                return False

            # Create message
            msg = MIMEMultipart()
            msg["From"] = from_email
            msg["To"] = ", ".join(recipients)
            msg["Subject"] = f"[{severity.upper()}] {title}"

            body = f"""
Disaster Recovery Alert

Title: {title}
Severity: {severity.upper()}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Message:
{message}

---
Pynomaly Disaster Recovery System
            """

            msg.attach(MIMEText(body, "plain"))

            # Send email
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(username, password)
                server.send_message(msg)

            return True

        except Exception as e:
            self.logger.error(f"Failed to send email notification: {e}")
            return False


class SlackNotificationChannel(NotificationChannel):
    """Slack notification channel."""

    async def send_notification(
        self, title: str, message: str, severity: str = "info"
    ) -> bool:
        """Send Slack notification."""
        try:
            import aiohttp

            webhook_url = self.config.get("webhook_url")
            channel = self.config.get("channel", "#alerts")

            if not webhook_url:
                self.logger.warning("Slack webhook URL not configured")
                return False

            # Color based on severity
            color_map = {
                "critical": "#FF0000",
                "high": "#FF6600",
                "warning": "#FFCC00",
                "info": "#0066CC",
                "success": "#00CC00",
            }
            color = color_map.get(severity, "#808080")

            payload = {
                "channel": channel,
                "username": "Pynomaly DR Bot",
                "icon_emoji": ":warning:",
                "attachments": [
                    {
                        "color": color,
                        "title": title,
                        "text": message,
                        "fields": [
                            {
                                "title": "Severity",
                                "value": severity.upper(),
                                "short": True,
                            },
                            {
                                "title": "Time",
                                "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "short": True,
                            },
                        ],
                        "footer": "Pynomaly Disaster Recovery",
                        "ts": int(time.time()),
                    }
                ],
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload) as response:
                    return response.status == 200

        except Exception as e:
            self.logger.error(f"Failed to send Slack notification: {e}")
            return False


class WebhookNotificationChannel(NotificationChannel):
    """Generic webhook notification channel."""

    async def send_notification(
        self, title: str, message: str, severity: str = "info"
    ) -> bool:
        """Send webhook notification."""
        try:
            import aiohttp

            webhook_url = self.config.get("webhook_url")
            auth_header = self.config.get("auth_header")

            if not webhook_url:
                self.logger.warning("Webhook URL not configured")
                return False

            payload = {
                "title": title,
                "message": message,
                "severity": severity,
                "timestamp": datetime.now().isoformat(),
                "source": "pynomaly_disaster_recovery",
            }

            headers = {"Content-Type": "application/json"}
            if auth_header:
                headers["Authorization"] = auth_header

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    webhook_url, json=payload, headers=headers
                ) as response:
                    return response.status < 300

        except Exception as e:
            self.logger.error(f"Failed to send webhook notification: {e}")
            return False


class HealthChecker:
    """Service health checker."""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def check_service_health(
        self, service_name: str, health_check_url: str
    ) -> bool:
        """Check if service is healthy."""
        try:
            import aiohttp

            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(health_check_url) as response:
                    is_healthy = response.status < 400

                    if is_healthy:
                        self.logger.debug(f"Service {service_name} is healthy")
                    else:
                        self.logger.warning(
                            f"Service {service_name} health check failed: {response.status}"
                        )

                    return is_healthy

        except Exception as e:
            self.logger.error(f"Health check failed for {service_name}: {e}")
            return False

    async def check_database_health(self, db_config: dict[str, Any]) -> bool:
        """Check database connectivity."""
        try:
            db_type = db_config.get("type", "postgresql")

            if db_type == "postgresql":
                return await self._check_postgresql_health(db_config)
            elif db_type == "mysql":
                return await self._check_mysql_health(db_config)
            elif db_type == "mongodb":
                return await self._check_mongodb_health(db_config)
            else:
                self.logger.warning(f"Unknown database type: {db_type}")
                return False

        except Exception as e:
            self.logger.error(f"Database health check failed: {e}")
            return False

    async def _check_postgresql_health(self, config: dict[str, Any]) -> bool:
        """Check PostgreSQL health."""
        try:
            import asyncpg

            connection_string = (
                f"postgresql://{config.get('username')}:{config.get('password')}"
                f"@{config.get('host', 'localhost')}:{config.get('port', 5432)}"
                f"/{config.get('database')}"
            )

            conn = await asyncpg.connect(connection_string, timeout=10)
            await conn.execute("SELECT 1")
            await conn.close()
            return True

        except Exception as e:
            self.logger.error(f"PostgreSQL health check failed: {e}")
            return False

    async def _check_mysql_health(self, config: dict[str, Any]) -> bool:
        """Check MySQL health."""
        try:
            import aiomysql

            conn = await aiomysql.connect(
                host=config.get("host", "localhost"),
                port=config.get("port", 3306),
                user=config.get("username"),
                password=config.get("password"),
                db=config.get("database"),
                connect_timeout=10,
            )

            cursor = await conn.cursor()
            await cursor.execute("SELECT 1")
            await cursor.close()
            conn.close()
            return True

        except Exception as e:
            self.logger.error(f"MySQL health check failed: {e}")
            return False

    async def _check_mongodb_health(self, config: dict[str, Any]) -> bool:
        """Check MongoDB health."""
        try:
            import motor.motor_asyncio

            connection_string = (
                f"mongodb://{config.get('username')}:{config.get('password')}"
                f"@{config.get('host', 'localhost')}:{config.get('port', 27017)}"
                f"/{config.get('database')}"
            )

            client = motor.motor_asyncio.AsyncIOMotorClient(
                connection_string, serverSelectionTimeoutMS=10000
            )

            await client.admin.command("ping")
            client.close()
            return True

        except Exception as e:
            self.logger.error(f"MongoDB health check failed: {e}")
            return False


class DisasterRecoveryService:
    """Main disaster recovery service."""

    def __init__(self, config_path: str = "config/disaster_recovery/dr_config.yml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Initialize components
        self.backup_manager = BackupManager()
        self.health_checker = HealthChecker()

        # Recovery tracking
        self.recovery_plans: dict[str, RecoveryPlan] = {}
        self.active_operations: dict[str, RecoveryOperation] = {}
        self.operation_history: list[RecoveryOperation] = []

        # Notification channels
        self.notification_channels: dict[str, NotificationChannel] = {}

        # Initialize from config
        self._initialize_recovery_plans()
        self._initialize_notification_channels()

        # Background tasks
        self._monitoring_tasks: list[asyncio.Task] = []
        self._running = False

    def _load_config(self) -> dict[str, Any]:
        """Load disaster recovery configuration."""
        if self.config_path.exists():
            with open(self.config_path) as f:
                return yaml.safe_load(f) or {}

        # Default configuration
        return {
            "monitoring": {
                "enabled": True,
                "check_interval_minutes": 5,
                "failure_threshold": 3,
            },
            "notification_channels": {},
            "recovery_plans": {},
        }

    def _initialize_recovery_plans(self):
        """Initialize recovery plans from configuration."""
        plans_config = self.config.get("recovery_plans", {})

        for plan_id, plan_config in plans_config.items():
            recovery_points = []

            for rp_config in plan_config.get("recovery_points", []):
                recovery_point = RecoveryPoint(
                    service_name=rp_config["service_name"],
                    rpo_minutes=rp_config.get("rpo_minutes", 60),
                    rto_minutes=rp_config.get("rto_minutes", 120),
                    priority=RecoveryPriority(rp_config.get("priority", "medium")),
                    backup_frequency_minutes=rp_config.get(
                        "backup_frequency_minutes", 60
                    ),
                    dependencies=rp_config.get("dependencies", []),
                    health_check_url=rp_config.get("health_check_url"),
                    restart_command=rp_config.get("restart_command"),
                )
                recovery_points.append(recovery_point)

            plan = RecoveryPlan(
                plan_id=plan_id,
                name=plan_config["name"],
                description=plan_config.get("description", ""),
                recovery_points=recovery_points,
                notification_channels=plan_config.get("notification_channels", []),
                pre_recovery_scripts=plan_config.get("pre_recovery_scripts", []),
                post_recovery_scripts=plan_config.get("post_recovery_scripts", []),
                estimated_rto_minutes=plan_config.get("estimated_rto_minutes", 0),
            )

            self.recovery_plans[plan_id] = plan

    def _initialize_notification_channels(self):
        """Initialize notification channels."""
        channels_config = self.config.get("notification_channels", {})

        for channel_name, channel_config in channels_config.items():
            channel_type = channel_config.get("type")

            if channel_type == "email":
                channel = EmailNotificationChannel(channel_name, channel_config)
            elif channel_type == "slack":
                channel = SlackNotificationChannel(channel_name, channel_config)
            elif channel_type == "webhook":
                channel = WebhookNotificationChannel(channel_name, channel_config)
            else:
                self.logger.warning(
                    f"Unknown notification channel type: {channel_type}"
                )
                continue

            self.notification_channels[channel_name] = channel

    async def start_monitoring(self):
        """Start disaster recovery monitoring."""
        if self._running:
            return

        self._running = True
        monitoring_config = self.config.get("monitoring", {})

        if monitoring_config.get("enabled", True):
            check_interval = monitoring_config.get("check_interval_minutes", 5) * 60

            self._monitoring_tasks.append(
                asyncio.create_task(self._monitor_services(check_interval))
            )
            self._monitoring_tasks.append(
                asyncio.create_task(self._automated_backup_scheduler())
            )

        self.logger.info("Disaster recovery monitoring started")

    async def stop_monitoring(self):
        """Stop disaster recovery monitoring."""
        self._running = False

        for task in self._monitoring_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        self._monitoring_tasks.clear()
        self.logger.info("Disaster recovery monitoring stopped")

    async def _monitor_services(self, check_interval: int):
        """Monitor service health and trigger recovery if needed."""
        failure_counts: dict[str, int] = {}
        failure_threshold = self.config.get("monitoring", {}).get(
            "failure_threshold", 3
        )

        while self._running:
            try:
                for plan in self.recovery_plans.values():
                    for recovery_point in plan.recovery_points:
                        service_name = recovery_point.service_name

                        if recovery_point.health_check_url:
                            is_healthy = await self.health_checker.check_service_health(
                                service_name, recovery_point.health_check_url
                            )

                            if is_healthy:
                                failure_counts.pop(service_name, 0)
                            else:
                                failure_counts[service_name] = (
                                    failure_counts.get(service_name, 0) + 1
                                )

                                if failure_counts[service_name] >= failure_threshold:
                                    await self._handle_service_failure(
                                        plan.plan_id, service_name
                                    )
                                    failure_counts[service_name] = 0

                await asyncio.sleep(check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Service monitoring error: {e}")
                await asyncio.sleep(60)

    async def _automated_backup_scheduler(self):
        """Schedule automated backups based on recovery points."""
        last_backup_times: dict[str, datetime] = {}

        while self._running:
            try:
                current_time = datetime.now()

                for plan in self.recovery_plans.values():
                    for recovery_point in plan.recovery_points:
                        service_name = recovery_point.service_name
                        backup_key = f"{plan.plan_id}_{service_name}"

                        last_backup = last_backup_times.get(backup_key)
                        frequency_delta = timedelta(
                            minutes=recovery_point.backup_frequency_minutes
                        )

                        if (
                            not last_backup
                            or (current_time - last_backup) >= frequency_delta
                        ):
                            await self._create_automated_backup(
                                plan.plan_id, service_name
                            )
                            last_backup_times[backup_key] = current_time

                await asyncio.sleep(300)  # Check every 5 minutes

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Backup scheduling error: {e}")
                await asyncio.sleep(600)

    async def _handle_service_failure(self, plan_id: str, service_name: str):
        """Handle service failure detection."""
        self.logger.error(f"Service failure detected: {service_name} in plan {plan_id}")

        await self._send_notification(
            title=f"Service Failure Detected: {service_name}",
            message=f"Service {service_name} in recovery plan {plan_id} has failed health checks. "
            f"Automatic recovery will be attempted.",
            severity="critical",
            plan_id=plan_id,
        )

        # Trigger automatic recovery
        await self.execute_recovery_plan(
            plan_id, triggered_by="automatic_failure_detection"
        )

    async def _create_automated_backup(self, plan_id: str, service_name: str):
        """Create automated backup for service."""
        try:
            backup_name = f"{plan_id}_{service_name}_auto"

            # Determine backup source and type based on service
            # This would need to be configured per service
            source_path = f"/var/lib/{service_name}"  # Example path

            backup_id = await self.backup_manager.create_backup(
                backup_name=backup_name,
                source_type="directory",
                source_path=source_path,
                backup_type=BackupType.INCREMENTAL,
                tags={"plan_id": plan_id, "service": service_name, "automated": "true"},
            )

            self.logger.info(
                f"Automated backup created for {service_name}: {backup_id}"
            )

        except Exception as e:
            self.logger.error(f"Automated backup failed for {service_name}: {e}")

    async def execute_recovery_plan(
        self, plan_id: str, triggered_by: str = "manual"
    ) -> str:
        """Execute disaster recovery plan."""
        if plan_id not in self.recovery_plans:
            raise ValueError(f"Recovery plan not found: {plan_id}")

        plan = self.recovery_plans[plan_id]
        operation_id = f"{plan_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        operation = RecoveryOperation(
            operation_id=operation_id,
            plan_id=plan_id,
            status=RecoveryStatus.PENDING,
            start_time=datetime.now(),
            triggered_by=triggered_by,
        )

        self.active_operations[operation_id] = operation

        try:
            operation.status = RecoveryStatus.RUNNING

            await self._send_notification(
                title=f"Disaster Recovery Started: {plan.name}",
                message=f"Recovery plan '{plan.name}' has been initiated. Operation ID: {operation_id}",
                severity="warning",
                plan_id=plan_id,
            )

            # Execute pre-recovery scripts
            for script in plan.pre_recovery_scripts:
                await self._execute_script(script, "pre-recovery")

            # Sort recovery points by priority
            sorted_points = sorted(
                plan.recovery_points,
                key=lambda rp: list(RecoveryPriority).index(rp.priority),
            )

            # Execute recovery for each service
            for recovery_point in sorted_points:
                success = await self._recover_service(recovery_point, operation)

                if success:
                    operation.recovered_services.append(recovery_point.service_name)
                else:
                    operation.failed_services.append(recovery_point.service_name)

            # Execute post-recovery scripts
            for script in plan.post_recovery_scripts:
                await self._execute_script(script, "post-recovery")

            # Determine final status
            if operation.failed_services:
                operation.status = (
                    RecoveryStatus.PARTIAL
                    if operation.recovered_services
                    else RecoveryStatus.FAILED
                )
            else:
                operation.status = RecoveryStatus.COMPLETED

            operation.end_time = datetime.now()

            # Send completion notification
            severity = (
                "success" if operation.status == RecoveryStatus.COMPLETED else "warning"
            )
            await self._send_notification(
                title=f"Disaster Recovery {'Completed' if operation.status == RecoveryStatus.COMPLETED else 'Partially Completed'}",
                message=f"Recovery operation {operation_id} finished. "
                f"Recovered: {len(operation.recovered_services)}, "
                f"Failed: {len(operation.failed_services)}",
                severity=severity,
                plan_id=plan_id,
            )

        except Exception as e:
            operation.status = RecoveryStatus.FAILED
            operation.error_message = str(e)
            operation.end_time = datetime.now()

            await self._send_notification(
                title=f"Disaster Recovery Failed: {plan.name}",
                message=f"Recovery operation {operation_id} failed: {str(e)}",
                severity="critical",
                plan_id=plan_id,
            )

            self.logger.error(f"Recovery plan execution failed: {e}")

        finally:
            # Move to history
            self.operation_history.append(operation)
            if operation_id in self.active_operations:
                del self.active_operations[operation_id]

        return operation_id

    async def _recover_service(
        self, recovery_point: RecoveryPoint, operation: RecoveryOperation
    ) -> bool:
        """Recover individual service."""
        service_name = recovery_point.service_name

        try:
            self.logger.info(f"Starting recovery for service: {service_name}")

            # Check dependencies first
            for dependency in recovery_point.dependencies:
                if dependency not in operation.recovered_services:
                    self.logger.warning(
                        f"Dependency {dependency} not yet recovered for {service_name}"
                    )
                    return False

            # Stop service if restart command is provided
            if recovery_point.restart_command:
                stop_command = recovery_point.restart_command.replace("start", "stop")
                await self._execute_command(stop_command, f"stop_{service_name}")

            # Find latest backup for this service
            backups = await self.backup_manager.list_backups()
            service_backups = [
                b
                for b in backups
                if b.get("tags", {}).get("service") == service_name
                and b.get("status") == BackupStatus.COMPLETED.value
            ]

            if service_backups:
                # Get most recent backup
                latest_backup = max(service_backups, key=lambda b: b["timestamp"])
                backup_id = latest_backup["backup_id"]

                # Restore from backup
                restore_path = f"/var/lib/{service_name}_restore"
                restore_success = await self.backup_manager.restore_backup(
                    backup_id, restore_path
                )

                if not restore_success:
                    self.logger.error(f"Failed to restore backup for {service_name}")
                    return False

            # Restart service
            if recovery_point.restart_command:
                restart_success = await self._execute_command(
                    recovery_point.restart_command, f"restart_{service_name}"
                )

                if not restart_success:
                    return False

            # Wait for service to become healthy
            if recovery_point.health_check_url:
                max_attempts = 30  # 5 minutes with 10-second intervals
                for attempt in range(max_attempts):
                    if await self.health_checker.check_service_health(
                        service_name, recovery_point.health_check_url
                    ):
                        self.logger.info(
                            f"Service {service_name} is healthy after recovery"
                        )
                        return True

                    await asyncio.sleep(10)

                self.logger.error(
                    f"Service {service_name} did not become healthy after recovery"
                )
                return False

            return True

        except Exception as e:
            self.logger.error(f"Service recovery failed for {service_name}: {e}")
            return False

    async def _execute_script(self, script_path: str, script_type: str):
        """Execute recovery script."""
        try:
            self.logger.info(f"Executing {script_type} script: {script_path}")

            result = subprocess.run(
                ["/bin/bash", script_path],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutes timeout
            )

            if result.returncode == 0:
                self.logger.info(f"{script_type} script completed successfully")
            else:
                self.logger.error(f"{script_type} script failed: {result.stderr}")

        except Exception as e:
            self.logger.error(f"Failed to execute {script_type} script: {e}")

    async def _execute_command(self, command: str, operation_name: str) -> bool:
        """Execute system command."""
        try:
            self.logger.info(f"Executing {operation_name}: {command}")

            result = subprocess.run(
                command.split(), capture_output=True, text=True, timeout=120
            )

            if result.returncode == 0:
                self.logger.info(f"{operation_name} completed successfully")
                return True
            else:
                self.logger.error(f"{operation_name} failed: {result.stderr}")
                return False

        except Exception as e:
            self.logger.error(f"Failed to execute {operation_name}: {e}")
            return False

    async def _send_notification(
        self, title: str, message: str, severity: str, plan_id: str
    ):
        """Send notification through configured channels."""
        plan = self.recovery_plans.get(plan_id)
        if not plan:
            return

        for channel_name in plan.notification_channels:
            channel = self.notification_channels.get(channel_name)
            if channel:
                try:
                    await channel.send_notification(title, message, severity)
                except Exception as e:
                    self.logger.error(
                        f"Failed to send notification via {channel_name}: {e}"
                    )

    async def test_recovery_plan(self, plan_id: str) -> dict[str, Any]:
        """Test disaster recovery plan without actually executing recovery."""
        if plan_id not in self.recovery_plans:
            raise ValueError(f"Recovery plan not found: {plan_id}")

        plan = self.recovery_plans[plan_id]
        test_results = {
            "plan_id": plan_id,
            "plan_name": plan.name,
            "test_time": datetime.now().isoformat(),
            "overall_status": "passed",
            "service_tests": [],
            "backup_tests": [],
            "script_tests": [],
        }

        # Test service health checks
        for recovery_point in plan.recovery_points:
            service_test = {
                "service_name": recovery_point.service_name,
                "health_check_passed": False,
                "backup_available": False,
                "dependencies_met": True,
            }

            # Test health check
            if recovery_point.health_check_url:
                service_test[
                    "health_check_passed"
                ] = await self.health_checker.check_service_health(
                    recovery_point.service_name, recovery_point.health_check_url
                )
            else:
                service_test["health_check_passed"] = True

            # Check backup availability
            backups = await self.backup_manager.list_backups()
            service_backups = [
                b
                for b in backups
                if b.get("tags", {}).get("service") == recovery_point.service_name
            ]
            service_test["backup_available"] = len(service_backups) > 0

            # Check dependencies
            for dependency in recovery_point.dependencies:
                dep_healthy = any(
                    rp.service_name == dependency and rp.health_check_url
                    for rp in plan.recovery_points
                )
                if not dep_healthy:
                    service_test["dependencies_met"] = False

            test_results["service_tests"].append(service_test)

            if not all(
                [
                    service_test["health_check_passed"],
                    service_test["backup_available"],
                    service_test["dependencies_met"],
                ]
            ):
                test_results["overall_status"] = "failed"

        # Test script accessibility
        for script in plan.pre_recovery_scripts + plan.post_recovery_scripts:
            script_test = {
                "script_path": script,
                "accessible": Path(script).exists(),
                "executable": Path(script).is_file() and os.access(script, os.X_OK),
            }
            test_results["script_tests"].append(script_test)

            if not script_test["accessible"] or not script_test["executable"]:
                test_results["overall_status"] = "failed"

        # Update last tested time
        plan.last_tested = datetime.now()

        return test_results

    def get_recovery_status(self) -> dict[str, Any]:
        """Get overall disaster recovery status."""
        total_operations = len(self.operation_history)
        successful_operations = len(
            [
                op
                for op in self.operation_history
                if op.status == RecoveryStatus.COMPLETED
            ]
        )

        return {
            "monitoring_active": self._running,
            "recovery_plans": len(self.recovery_plans),
            "active_operations": len(self.active_operations),
            "total_operations": total_operations,
            "successful_operations": successful_operations,
            "success_rate": (successful_operations / total_operations * 100)
            if total_operations > 0
            else 0,
            "notification_channels": len(self.notification_channels),
            "last_operation": self.operation_history[-1].to_dict()
            if self.operation_history
            else None,
        }

    def export_recovery_plan(self, plan_id: str) -> dict[str, Any]:
        """Export recovery plan configuration."""
        if plan_id not in self.recovery_plans:
            raise ValueError(f"Recovery plan not found: {plan_id}")

        return self.recovery_plans[plan_id].to_dict()

    async def shutdown(self):
        """Shutdown disaster recovery service."""
        await self.stop_monitoring()
        self.logger.info("Disaster recovery service shutdown complete")


# Factory function
def create_disaster_recovery_service(
    config_path: str = None,
) -> DisasterRecoveryService:
    """Create disaster recovery service with configuration."""
    return DisasterRecoveryService(
        config_path or "config/disaster_recovery/dr_config.yml"
    )


if __name__ == "__main__":

    async def main():
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        dr_service = create_disaster_recovery_service()
        await dr_service.start_monitoring()

        # Keep running
        try:
            while True:
                await asyncio.sleep(60)
        except KeyboardInterrupt:
            await dr_service.shutdown()

    asyncio.run(main())
