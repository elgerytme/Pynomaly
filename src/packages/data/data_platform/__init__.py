"""
Pynomaly Data Platform Package

Unified data processing, profiling, quality, and science package providing:
- Data transformation and cleaning pipelines
- Statistical profiling and schema analysis
- Quality validation and compliance
- Data science and analytics capabilities

This package consolidates all data-related functionality into a cohesive platform.
"""

__version__ = "1.0.0"
__author__ = "Pynomaly Team"

# Re-export main components for convenience
from .transformation import *
from .profiling import *
from .quality import *
from .science import *