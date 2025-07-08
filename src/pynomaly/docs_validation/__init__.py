"""Documentation validation framework for Pynomaly.

This package provides comprehensive documentation validation tools to ensure
documentation quality, consistency, and accuracy across the project.
"""

from .checkers.consistency import ConsistencyChecker
from .checkers.content import ContentChecker
from .checkers.links import LinkChecker
from .checkers.structure import StructureChecker
from .core.config import ValidationConfig
from .core.reporter import ValidationReporter
from .core.validator import DocumentationValidator

__version__ = "1.0.0"
__author__ = "Pynomaly Team"

__all__ = [
    "DocumentationValidator",
    "ValidationConfig",
    "ValidationReporter",
    "ContentChecker",
    "StructureChecker",
    "LinkChecker",
    "ConsistencyChecker",
]
