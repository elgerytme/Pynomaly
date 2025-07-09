"""
Test orchestration module for coordinating different types of testing.

This module provides the orchestrator that combines:
- Traditional pytest runs
- Property-based testing with Hypothesis
- Mutation testing
- Comprehensive reporting
"""

from .orchestrator import TestOrchestrator, main

__version__ = "0.1.0"
__all__ = ["TestOrchestrator", "main"]
