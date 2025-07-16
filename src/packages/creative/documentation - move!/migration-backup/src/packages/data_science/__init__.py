"""Data Science Package - Comprehensive Domain Model Implementation.

This package provides comprehensive data science capabilities including:
- Statistical analysis and profiling
- Machine learning pipeline management  
- Feature engineering and selection
- Exploratory data analysis
- Advanced analytics and insights
- Model lifecycle management
- Experiment tracking and management

The package follows clean architecture principles with domain-driven design.
"""

__version__ = "0.1.0"

# Domain Entities
from .domain.entities.data_science_model import DataScienceModel, ModelType, ModelStatus
from .domain.entities.experiment import Experiment
from .domain.entities.feature_store import FeatureStore, FeatureType, FeatureStatus
from .domain.entities.dataset_profile import DatasetProfile
from .domain.entities.analysis_job import AnalysisJob
from .domain.entities.statistical_profile import StatisticalProfile
from .domain.entities.machine_learning_pipeline import MachineLearningPipeline

# Domain Value Objects
from .domain.value_objects.statistical_metrics import StatisticalMetrics
from .domain.value_objects.model_performance_metrics import ModelPerformanceMetrics
from .domain.value_objects.performance_degradation_metrics import PerformanceDegradationMetrics
from .domain.value_objects.ml_model_metrics import MLModelMetrics
from .domain.value_objects.feature_importance import FeatureImportance

# Domain Services
from .domain.services.statistical_analysis_service import StatisticalAnalysisService
from .domain.services.model_training_service import ModelTrainingService
from .domain.services.model_lifecycle_service import ModelLifecycleService

# Infrastructure
from .infrastructure.config.data_science_settings import DataScienceSettings
from .infrastructure.di.container import DataScienceContainer

__all__ = [
    # Core entities
    "DataScienceModel",
    "Experiment",
    "FeatureStore", 
    "DatasetProfile",
    "AnalysisJob",
    "StatisticalProfile",
    "MachineLearningPipeline",
    # Enums
    "ModelType",
    "ModelStatus", 
    "FeatureType",
    "FeatureStatus",
    # Value objects
    "StatisticalMetrics",
    "ModelPerformanceMetrics",
    "PerformanceDegradationMetrics",
    "MLModelMetrics",
    "FeatureImportance",
    # Domain services
    "StatisticalAnalysisService",
    "ModelTrainingService", 
    "ModelLifecycleService",
    # Infrastructure
    "DataScienceSettings",
    "DataScienceContainer",
]