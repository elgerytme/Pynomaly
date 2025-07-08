"""
Domain optimization module for Pynomaly.

This module provides optimization utilities for multi-objective optimization,
including Pareto optimization for model selection.
"""

from .pareto_optimizer import ParetoOptimizer

__all__ = ["ParetoOptimizer"]
