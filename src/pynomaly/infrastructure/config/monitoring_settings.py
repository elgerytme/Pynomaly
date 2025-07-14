"""Monitoring and observability configuration settings."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class MonitoringSettings(BaseModel):
    """Monitoring and observability settings."""

    metrics_enabled: bool = True
    tracing_enabled: bool = False
    prometheus_enabled: bool = True
    prometheus_port: int = 9090
    otlp_endpoint: str | None = None
    otlp_insecure: bool = True
    log_level: str = "INFO"
    log_format: str = "json"
    host_name: str = "localhost"
    instrument_fastapi: bool = True
    instrument_sqlalchemy: bool = True

    def get_logging_config(self) -> dict[str, Any]:
        """Get logging configuration."""
        return {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": {
                    "()": "structlog.stdlib.ProcessorFormatter",
                    "processor": "structlog.processors.JSONRenderer()",
                },
                "text": {"format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"},
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": self.log_format,
                    "stream": "ext://sys.stdout",
                }
            },
            "root": {"level": self.log_level, "handlers": ["console"]},
        }
