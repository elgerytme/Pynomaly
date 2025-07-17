"""Configuration management for comprehensive static analysis."""

from .manager import ConfigManager, AnalysisConfig
from .profiles import ANALYSIS_PROFILES

__all__ = ["ConfigManager", "AnalysisConfig", "ANALYSIS_PROFILES"]