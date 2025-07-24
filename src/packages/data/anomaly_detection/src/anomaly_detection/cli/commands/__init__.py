"""CLI commands package."""

# Import all command modules to ensure they're available
from . import detection, models, worker, streaming, explain, batch, optimize, reports, monitor

__all__ = [
    "detection",
    "models", 
    "worker",
    "streaming",
    "explain",
    "batch",
    "optimize",
    "reports",
    "monitor"
]