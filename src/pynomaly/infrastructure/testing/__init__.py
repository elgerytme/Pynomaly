"""
Testing infrastructure for algorithm comparison and performance evaluation.

This module provides comprehensive testing capabilities including:
- A/B testing framework for comparing algorithms
- Statistical analysis and significance testing
- Performance benchmarking and metrics collection
- Traffic splitting and experiment management
"""

from .ab_testing_framework import (
    ABTest,
    ABTestingService,
    TestVariant,
    TestStatus,
    SplitStrategy,
    MetricType,
    TestMetric,
    TestResult,
    StatisticalResult,
    TrafficSplitter,
    MetricsCalculator
)

__all__ = [
    # Core testing classes
    "ABTest",
    "ABTestingService",
    "TestVariant",
    "TestResult",
    "TestMetric",
    "StatisticalResult",
    
    # Enums
    "TestStatus",
    "SplitStrategy", 
    "MetricType",
    
    # Utilities
    "TrafficSplitter",
    "MetricsCalculator"
]