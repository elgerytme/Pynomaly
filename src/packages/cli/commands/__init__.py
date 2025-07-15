"""Data Science CLI Commands Package.

This package provides comprehensive command-line interface tools for data science operations,
including data profiling, quality management, analytics, and pipeline automation.
"""

from .data_science import data_science_app
from .profiling import profiling_app
from .quality import quality_app
from .analytics import analytics_app
from .config import config_app
from .pipeline import pipeline_app

__all__ = [
    "data_science_app",
    "profiling_app", 
    "quality_app",
    "analytics_app",
    "config_app",
    "pipeline_app"
]