"""Scheduler entities for job management."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ResourceRequirement:
    """Resource requirements for a job."""

    cpu_cores: float = 1.0
    memory_gb: float = 2.0
    workers: int = 1
    gpu_count: int = 0


@dataclass
class JobDefinition:
    """Definition of a scheduled job."""

    job_id: str
    name: str
    depends_on: set[str] = field(default_factory=set)
    resource_requirements: ResourceRequirement = field(default_factory=ResourceRequirement)
    cron_expression: str | None = None
    interval_seconds: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self) -> None:
        """Validate job definition after initialization."""
        if not self.job_id:
            raise ValueError("Job ID cannot be empty")
        if not self.name:
            raise ValueError("Job name cannot be empty")


@dataclass
class JobExecution:
    """Record of a job execution."""

    execution_id: str
    job_id: str
    started_at: datetime
    finished_at: datetime | None = None
    status: str = "running"  # running, completed, failed
    result: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None
