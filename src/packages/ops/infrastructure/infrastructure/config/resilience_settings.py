"""Resilience and retry configuration settings."""

from __future__ import annotations

from pydantic import BaseModel


class ResilienceSettings(BaseModel):
    """Resilience and retry configuration settings."""

    # Resilience settings
    resilience_enabled: bool = True
    default_timeout: float = 5.0
    database_timeout: float = 30.0
    api_timeout: float = 60.0
    cache_timeout: float = 5.0
    file_timeout: float = 10.0
    ml_timeout: float = 300.0

    # Retry settings
    default_max_attempts: int = 3
    database_max_attempts: int = 3
    api_max_attempts: int = 5
    cache_max_attempts: int = 2
    file_max_attempts: int = 3
    ml_max_attempts: int = 2

    # Retry backoff settings
    default_base_delay: float = 1.0
    default_max_delay: float = 60.0
    default_exponential_base: float = 2.0
    default_jitter: bool = True
    database_base_delay: float = 0.5
    database_max_delay: float = 10.0
    api_base_delay: float = 1.0
    api_max_delay: float = 30.0
    cache_base_delay: float = 0.1
    cache_max_delay: float = 1.0
    file_base_delay: float = 0.2
    file_max_delay: float = 5.0
    ml_base_delay: float = 5.0
    ml_max_delay: float = 30.0

    # Circuit breaker settings
    default_failure_threshold: int = 5
    default_recovery_timeout: float = 60.0
    database_failure_threshold: int = 3
    database_recovery_timeout: float = 30.0
    api_failure_threshold: int = 5
    api_recovery_timeout: float = 60.0
    cache_failure_threshold: int = 3
    cache_recovery_timeout: float = 15.0
    file_failure_threshold: int = 3
    file_recovery_timeout: float = 30.0
    ml_failure_threshold: int = 2
    ml_recovery_timeout: float = 120.0
