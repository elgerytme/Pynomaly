"""Documentation validation framework for Pynomaly.

This package provides comprehensive documentation validation tools to ensure
documentation quality, consistency, and accuracy across the project.
"""

from .core.config import ValidationConfig
from .core.reporter import ValidationReporter
from .core.validator import DocumentationValidator

__version__ = "1.0.0"
__author__ = "Pynomaly Team"

__all__ = [
    "DocumentationValidator",
    "ValidationConfig",
    "ValidationReporter",
]
