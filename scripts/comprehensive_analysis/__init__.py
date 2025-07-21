"""
Comprehensive Static Analysis System

A compiler-level static analysis system for Python that provides:
- Advanced type checking and inference
- Security vulnerability scanning
- Performance analysis and optimization
- Documentation coverage analysis
- Automated code fixes

This system integrates multiple analysis tools to provide comprehensive
code quality assessment similar to advanced compilers like Rust, Haskell, etc.
"""

__version__ = "0.1.0"
__author__ = "anomaly_detection Development Team"

from .orchestrator import AnalysisOrchestrator
from .config.manager import ConfigManager, AnalysisConfig

__all__ = [
    "AnalysisOrchestrator",
    "ConfigManager", 
    "AnalysisConfig",
]