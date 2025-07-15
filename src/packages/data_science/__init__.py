"""Data Science Package - Domain Model Implementation.

This package provides comprehensive data science capabilities including:
- Statistical analysis and profiling
- Machine learning pipeline management
- Feature engineering and selection
- Exploratory data analysis
- Advanced analytics and insights

The package follows clean architecture principles with domain-driven design.
"""

__version__ = "0.1.0"

from .domain.entities import (
    DataScienceModel,
    AnalysisJob,
    StatisticalProfile,
    MachineLearningPipeline,
    FeatureStore,
)

from .domain.value_objects import (
    StatisticalMetrics,
    ModelPerformanceMetrics,
    PerformanceDegradationMetrics,
)

__all__ = [
    # Core entities
    "DataScienceModel",
    "AnalysisJob", 
    "StatisticalProfile",
    "MachineLearningPipeline",
    "FeatureStore",
    # Value objects
    "StatisticalMetrics",
    "ModelPerformanceMetrics",
    "PerformanceDegradationMetrics",
]