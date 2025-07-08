"""Mock utilities for fast and reliable testing.

This module provides comprehensive mocking for heavy operations including:
- ML model training and inference
- File I/O operations  
- Network requests
- Database operations
- External dependencies

The mocks are designed to be:
- Fast (no actual computation)
- Deterministic (consistent results)
- Lightweight (minimal memory usage)
- Platform-independent (no external dependencies)
"""

from .ml_mocks import (
    MockMLFrameworks,
    MockModelTraining,
    MockHyperparameterOptimization,
    MockAutoML,
)
from .io_mocks import (
    MockFileOperations,
    MockNetworkOperations,
    MockDatabaseOperations,
)
from .data_mocks import (
    MockDatasetRegistry,
    MockDetectorRegistry,
    create_lightweight_dataset,
    create_lightweight_detector,
)

__all__ = [
    "MockMLFrameworks",
    "MockModelTraining", 
    "MockHyperparameterOptimization",
    "MockAutoML",
    "MockFileOperations",
    "MockNetworkOperations",
    "MockDatabaseOperations",
    "MockDatasetRegistry",
    "MockDetectorRegistry",
    "create_lightweight_dataset",
    "create_lightweight_detector",
]
