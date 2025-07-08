"""Trigger manager module for handling cron and interval triggers."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Dict

try:
    from croniter import croniter
except ImportError:
    croniter = None


logger = logging.getLogger(__name__)


class TriggerType(Enum):
    """Type of trigger."""
    CRON = "cron"
    INTERVAL = "interval"
    MANUAL = "manual"


class Trigger(ABC):
    """Base class for triggers."""
    
    @abstractmethod
    def get_next_run_time(self, current_time: datetime) -> Optional[datetime]:
        """Get the next run time for this trigger."""
        pass
    
    @abstractmethod
    def should_run(self, current_time: datetime, last_run_time: Optional[datetime]) -> bool:
        """Check if the trigger should run at the current time."""
        pass


class CronTrigger(Trigger):
    """Cron-based trigger."""
    
    def __init__(self, cron_expression: str) -> None:
        """Initialize cron trigger."""
        if croniter is None:
            raise ImportError("croniter is required for cron triggers. Install it with: pip install croniter")
        
        self.cron_expression = cron_expression
        # Test if the cron expression is valid
        try:
            croniter(cron_expression)
        except ValueError as e:
            raise ValueError(f"Invalid cron expression '{cron_expression}': {e}")
    
    def get_next_run_time(self, current_time: datetime) -> Optional[datetime]:
        """Get the next run time based on the cron expression."""
        cron = croniter(self.cron_expression, current_time)
        return cron.get_next(datetime)
    
    def should_run(self, current_time: datetime, last_run_time: Optional[datetime]) -> bool:
        """Check if the trigger should run at the current time."""
        if last_run_time is None:
            return True
        
        # Get the next run time after the last run
        cron = croniter(self.cron_expression, last_run_time)
        next_run = cron.get_next(datetime)
        
        return current_time >= next_run


class IntervalTrigger(Trigger):
    """Interval-based trigger."""
    
    def __init__(self, interval_seconds: int) -> None:
        """Initialize interval trigger."""
        if interval_seconds <= 0:
            raise ValueError("Interval must be positive")
        
        self.interval_seconds = interval_seconds
        self.interval_timedelta = timedelta(seconds=interval_seconds)
    
    def get_next_run_time(self, current_time: datetime) -> Optional[datetime]:
        """Get the next run time based on the interval."""
        return current_time + self.interval_timedelta
    
    def should_run(self, current_time: datetime, last_run_time: Optional[datetime]) -> bool:
        """Check if the trigger should run at the current time."""
        if last_run_time is None:
            return True
        
        return current_time >= last_run_time + self.interval_timedelta


class ManualTrigger(Trigger):
    """Manual trigger (no automatic scheduling)."""
    
    def get_next_run_time(self, current_time: datetime) -> Optional[datetime]:
        """Manual triggers don't have a next run time."""
        return None
    
    def should_run(self, current_time: datetime, last_run_time: Optional[datetime]) -> bool:
        """Manual triggers don't run automatically."""
        return False


class TriggerManager:
    """Manages different types of triggers."""
    
    def __init__(self) -> None:
        """Initialize trigger manager."""
        self.triggers: Dict[str, Trigger] = {}
    
    def create_cron_trigger(self, schedule_id: str, cron_expression: str) -> CronTrigger:
        """Create a cron trigger."""
        trigger = CronTrigger(cron_expression)
        self.triggers[schedule_id] = trigger
        return trigger
    
    def create_interval_trigger(self, schedule_id: str, interval_seconds: int) -> IntervalTrigger:
        """Create an interval trigger."""
        trigger = IntervalTrigger(interval_seconds)
        self.triggers[schedule_id] = trigger
        return trigger
    
    def create_manual_trigger(self, schedule_id: str) -> ManualTrigger:
        """Create a manual trigger."""
        trigger = ManualTrigger()
        self.triggers[schedule_id] = trigger
        return trigger
    
    def get_trigger(self, schedule_id: str) -> Optional[Trigger]:
        """Get a trigger by schedule ID."""
        return self.triggers.get(schedule_id)
    
    def remove_trigger(self, schedule_id: str) -> None:
        """Remove a trigger."""
        self.triggers.pop(schedule_id, None)
    
    def should_run(self, schedule_id: str, current_time: datetime, last_run_time: Optional[datetime]) -> bool:
        """Check if a schedule should run."""
        trigger = self.get_trigger(schedule_id)
        if trigger is None:
            return False
        
        return trigger.should_run(current_time, last_run_time)
    
    def get_next_run_time(self, schedule_id: str, current_time: datetime) -> Optional[datetime]:
        """Get the next run time for a schedule."""
        trigger = self.get_trigger(schedule_id)
        if trigger is None:
            return None
        
        return trigger.get_next_run_time(current_time)
