"""
Structure Enforcer Package

This package provides validation and enforcement of project structure standards.
It serves as a reusable library for CLI tools, pre-commit hooks, and CI/CD pipelines.
"""

from .core import (
    Model,
    Fix,
    Violation,
    apply_fixes,
    detect_violations,
    scan_repository,
    suggest_fixes,
)

__all__ = [
    "Model",
    "Fix", 
    "Violation",
    "apply_fixes",
    "detect_violations",
    "scan_repository",
    "suggest_fixes",
]

__version__ = "1.0.0"
