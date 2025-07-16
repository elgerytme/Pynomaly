"""Trigger manager for the scheduler system."""

from __future__ import annotations

from datetime import datetime, timedelta

import croniter


class TriggerManager:
    """Manages cron/interval triggers for schedules."""

    @staticmethod
    def compute_next_execution(
        cron_expression: str | None = None,
        interval_seconds: int | None = None,
        current_time: datetime | None = None,
    ) -> datetime | None:
        """Compute the next execution time based on triggers."""
        current_time = current_time or datetime.now()

        if cron_expression:
            cron = croniter.croniter(cron_expression, current_time)
            return cron.get_next(datetime)

        if interval_seconds is not None:
            return current_time + timedelta(seconds=interval_seconds)

        return None

    @staticmethod
    def validate_cron_expression(cron_expression: str) -> bool:
        """Validate a cron expression."""
        try:
            croniter.croniter(cron_expression)
            return True
        except (ValueError, KeyError):
            return False
