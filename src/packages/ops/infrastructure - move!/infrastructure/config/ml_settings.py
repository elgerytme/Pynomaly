"""Machine learning configuration settings."""

from __future__ import annotations

from pydantic import BaseModel, field_validator


class MLSettings(BaseModel):
    """Machine learning configuration settings."""

    # Algorithm settings
    default_contamination_rate: float = 0.1
    max_parallel_detectors: int = 4
    detector_timeout: int = 300  # seconds

    # Data processing settings
    max_dataset_size_mb: int = 1000
    chunk_size: int = 10000
    max_features: int = 1000

    # ML settings
    random_seed: int = 42
    gpu_enabled: bool = False
    gpu_memory_fraction: float = 0.8

    # Streaming settings
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic_prefix: str = "monorepo"
    streaming_enabled: bool = False
    max_streaming_sessions: int = 10

    @field_validator("default_contamination_rate")
    @classmethod
    def validate_contamination_rate(cls, v: float) -> float:
        """Validate contamination rate is in valid range."""
        if not 0 <= v <= 1:
            raise ValueError("Contamination rate must be between 0 and 1")
        return v
