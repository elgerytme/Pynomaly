"""
Structure Enforcer Package

This package provides validation and enforcement of project structure standards.
It serves as a reusable library for CLI tools, pre-commit hooks, and CI/CD pipelines.
"""

from .core import (
    apply_fixes,
    detect_violations,
    scan_repository,
    suggest_fixes,
)
from .models import (
    Fix,
    FixResult,
    FixType,
    Model,
    Severity,
    Violation,
    ViolationType,
)

__all__ = [
    # Core functions
    "apply_fixes",
    "detect_violations",
    "scan_repository",
    "suggest_fixes",
    # Data models
    "Fix",
    "FixResult",
    "FixType",
    "Model",
    "Severity",
    "Violation",
    "ViolationType",
]

__version__ = "1.0.0"
