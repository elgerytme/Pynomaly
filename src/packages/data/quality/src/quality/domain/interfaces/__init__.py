"""
Domain interfaces for data quality operations.

This module contains interface definitions that define contracts for data quality services.
These interfaces follow domain-driven design patterns and provide abstraction layers
for various data quality operations.
"""

from .data_quality_interface import DataQualityInterface, QualityReport

__all__ = [
    'DataQualityInterface',
    'QualityReport',
]