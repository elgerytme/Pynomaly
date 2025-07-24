"""CLI commands package."""

# Import all command modules to ensure they're available
from . import detection, models, data, worker, streaming, explain, health, batch, optimize, reports, monitor

__all__ = [
    "detection",
    "models", 
    "data",
    "worker",
    "streaming",
    "explain",
    "health",
    "batch",
    "optimize",
    "reports",
    "monitor"
]