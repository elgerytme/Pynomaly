"""Data Science Domain Repository Interfaces.

Repository interfaces define contracts for data access without specifying implementation details.
"""

from .data_science_model_repository import IDataScienceModelRepository
from .analysis_job_repository import IAnalysisJobRepository
from .feature_store_repository import IFeatureStoreRepository
from .statistical_profile_repository import IStatisticalProfileRepository
from .machine_learning_pipeline_repository import IMachineLearningPipelineRepository
from .statistical_analysis_repository import StatisticalAnalysisRepository

__all__ = [
    "IDataScienceModelRepository",
    "IAnalysisJobRepository", 
    "IFeatureStoreRepository",
    "IStatisticalProfileRepository",
    "IMachineLearningPipelineRepository",
    "StatisticalAnalysisRepository",
]