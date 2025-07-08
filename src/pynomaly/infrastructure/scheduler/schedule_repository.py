"""Schedule repository for persisting schedule data."""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

from .entities import Schedule, ScheduleExecution, ExecutionResult


logger = logging.getLogger(__name__)


class ScheduleRepository(ABC):
    """Abstract base class for schedule repositories."""

    @abstractmethod
    def save_schedule(self, schedule: Schedule) -> None:
        """Save a schedule."""
        pass

    @abstractmethod
    def get_schedule(self, schedule_id: str) -> Optional[Schedule]:
        """Get a schedule by ID."""
        pass

    @abstractmethod
    def list_schedules(self) -> List[Schedule]:
        """List all schedules."""
        pass

    @abstractmethod
    def delete_schedule(self, schedule_id: str) -> bool:
        """Delete a schedule."""
        pass

    @abstractmethod
    def save_execution(self, execution: ScheduleExecution) -> None:
        """Save a schedule execution."""
        pass

    @abstractmethod
    def get_execution(self, execution_id: str) -> Optional[ScheduleExecution]:
        """Get a schedule execution by ID."""
        pass

    @abstractmethod
    def list_executions(
        self, schedule_id: Optional[str] = None
    ) -> List[ScheduleExecution]:
        """List schedule executions, optionally filtered by schedule ID."""
        pass

    @abstractmethod
    def delete_execution(self, execution_id: str) -> bool:
        """Delete a schedule execution."""
        pass


class InMemoryScheduleRepository(ScheduleRepository):
    """In-memory implementation of schedule repository."""

    def __init__(self) -> None:
        """Initialize in-memory repository."""
        self.schedules: Dict[str, Schedule] = {}
        self.executions: Dict[str, ScheduleExecution] = {}

    def save_schedule(self, schedule: Schedule) -> None:
        """Save a schedule."""
        self.schedules[schedule.schedule_id] = schedule
        logger.debug(f"Saved schedule {schedule.schedule_id}")

    def get_schedule(self, schedule_id: str) -> Optional[Schedule]:
        """Get a schedule by ID."""
        return self.schedules.get(schedule_id)

    def list_schedules(self) -> List[Schedule]:
        """List all schedules."""
        return list(self.schedules.values())

    def delete_schedule(self, schedule_id: str) -> bool:
        """Delete a schedule."""
        if schedule_id in self.schedules:
            del self.schedules[schedule_id]
            logger.debug(f"Deleted schedule {schedule_id}")
            return True
        return False

    def save_execution(self, execution: ScheduleExecution) -> None:
        """Save a schedule execution."""
        self.executions[execution.execution_id] = execution
        logger.debug(f"Saved execution {execution.execution_id}")

    def get_execution(self, execution_id: str) -> Optional[ScheduleExecution]:
        """Get a schedule execution by ID."""
        return self.executions.get(execution_id)

    def list_executions(
        self, schedule_id: Optional[str] = None
    ) -> List[ScheduleExecution]:
        """List schedule executions, optionally filtered by schedule ID."""
        executions = list(self.executions.values())
        if schedule_id:
            executions = [e for e in executions if e.schedule_id == schedule_id]
        return executions

    def delete_execution(self, execution_id: str) -> bool:
        """Delete a schedule execution."""
        if execution_id in self.executions:
            del self.executions[execution_id]
            logger.debug(f"Deleted execution {execution_id}")
            return True
        return False

    def clear_all(self) -> None:
        """Clear all data (for testing)."""
        self.schedules.clear()
        self.executions.clear()
        logger.debug("Cleared all data")


class FileSystemScheduleRepository(ScheduleRepository):
    """File system implementation of schedule repository."""

    def __init__(self, base_path: str = "data/schedules") -> None:
        """Initialize file system repository."""
        self.base_path = Path(base_path)
        self.schedules_path = self.base_path / "schedules"
        self.executions_path = self.base_path / "executions"

        # Create directories if they don't exist
        self.schedules_path.mkdir(parents=True, exist_ok=True)
        self.executions_path.mkdir(parents=True, exist_ok=True)

    def _schedule_file_path(self, schedule_id: str) -> Path:
        """Get file path for a schedule."""
        return self.schedules_path / f"{schedule_id}.json"

    def _execution_file_path(self, execution_id: str) -> Path:
        """Get file path for an execution."""
        return self.executions_path / f"{execution_id}.json"

    def save_schedule(self, schedule: Schedule) -> None:
        """Save a schedule."""
        file_path = self._schedule_file_path(schedule.schedule_id)
        try:
            with open(file_path, "w") as f:
                json.dump(schedule.to_dict(), f, indent=2)
            logger.debug(f"Saved schedule {schedule.schedule_id} to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save schedule {schedule.schedule_id}: {e}")
            raise

    def get_schedule(self, schedule_id: str) -> Optional[Schedule]:
        """Get a schedule by ID."""
        file_path = self._schedule_file_path(schedule_id)
        if not file_path.exists():
            return None

        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            return Schedule.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load schedule {schedule_id}: {e}")
            return None

    def list_schedules(self) -> List[Schedule]:
        """List all schedules."""
        schedules = []
        for file_path in self.schedules_path.glob("*.json"):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                schedule = Schedule.from_dict(data)
                schedules.append(schedule)
            except Exception as e:
                logger.error(f"Failed to load schedule from {file_path}: {e}")
        return schedules

    def delete_schedule(self, schedule_id: str) -> bool:
        """Delete a schedule."""
        file_path = self._schedule_file_path(schedule_id)
        if file_path.exists():
            try:
                file_path.unlink()
                logger.debug(f"Deleted schedule {schedule_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to delete schedule {schedule_id}: {e}")
        return False

    def save_execution(self, execution: ScheduleExecution) -> None:
        """Save a schedule execution."""
        file_path = self._execution_file_path(execution.execution_id)
        try:
            with open(file_path, "w") as f:
                json.dump(execution.to_dict(), f, indent=2)
            logger.debug(f"Saved execution {execution.execution_id} to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save execution {execution.execution_id}: {e}")
            raise

    def get_execution(self, execution_id: str) -> Optional[ScheduleExecution]:
        """Get a schedule execution by ID."""
        file_path = self._execution_file_path(execution_id)
        if not file_path.exists():
            return None

        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            # Reconstruct the execution
            execution = ScheduleExecution(
                execution_id=data["execution_id"],
                schedule_id=data["schedule_id"],
                trigger_type=data["trigger_type"],
                trigger_info=data.get("trigger_info", {}),
                created_at=data["created_at"],
            )

            # Add result if present
            if data.get("result"):
                result_data = data["result"]
                result = ExecutionResult(
                    execution_id=result_data["execution_id"],
                    schedule_id=result_data["schedule_id"],
                    status=result_data["status"],
                    started_at=result_data.get("started_at"),
                    completed_at=result_data.get("completed_at"),
                    job_instances=result_data.get("job_instances", {}),
                    jobs_total=result_data.get("jobs_total", 0),
                    jobs_completed=result_data.get("jobs_completed", 0),
                    jobs_failed=result_data.get("jobs_failed", 0),
                    jobs_cancelled=result_data.get("jobs_cancelled", 0),
                    trigger_type=result_data.get("trigger_type", "manual"),
                    trigger_info=result_data.get("trigger_info", {}),
                    metadata=result_data.get("metadata", {}),
                    created_at=result_data["created_at"],
                )
                execution.result = result

            return execution
        except Exception as e:
            logger.error(f"Failed to load execution {execution_id}: {e}")
            return None

    def list_executions(
        self, schedule_id: Optional[str] = None
    ) -> List[ScheduleExecution]:
        """List schedule executions, optionally filtered by schedule ID."""
        executions = []
        for file_path in self.executions_path.glob("*.json"):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)

                # Filter by schedule ID if provided
                if schedule_id and data.get("schedule_id") != schedule_id:
                    continue

                execution = ScheduleExecution(
                    execution_id=data["execution_id"],
                    schedule_id=data["schedule_id"],
                    trigger_type=data["trigger_type"],
                    trigger_info=data.get("trigger_info", {}),
                    created_at=data["created_at"],
                )
                executions.append(execution)
            except Exception as e:
                logger.error(f"Failed to load execution from {file_path}: {e}")
        return executions

    def delete_execution(self, execution_id: str) -> bool:
        """Delete a schedule execution."""
        file_path = self._execution_file_path(execution_id)
        if file_path.exists():
            try:
                file_path.unlink()
                logger.debug(f"Deleted execution {execution_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to delete execution {execution_id}: {e}")
        return False

    def clear_all(self) -> None:
        """Clear all data (for testing)."""
        import shutil

        if self.base_path.exists():
            shutil.rmtree(self.base_path)
        self.schedules_path.mkdir(parents=True, exist_ok=True)
        self.executions_path.mkdir(parents=True, exist_ok=True)
        logger.debug("Cleared all data")
