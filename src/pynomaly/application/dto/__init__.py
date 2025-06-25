"""Data Transfer Objects for application layer."""

from .detector_dto import DetectorDTO, CreateDetectorDTO, UpdateDetectorDTO, DetectorResponseDTO, DetectionRequestDTO
from .dataset_dto import DatasetDTO, CreateDatasetDTO, DataQualityReportDTO
from .result_dto import DetectionResultDTO, AnomalyDTO
from .experiment_dto import ExperimentDTO, RunDTO, CreateExperimentDTO, LeaderboardEntryDTO, ExperimentResponseDTO
from .automl_dto import (
    AutoMLRequestDTO, AutoMLResponseDTO, AutoMLResultDTO, DatasetProfileDTO,
    AlgorithmRecommendationDTO, HyperparameterOptimizationRequestDTO,
    HyperparameterOptimizationResponseDTO, AutoMLProfileRequestDTO,
    AutoMLProfileResponseDTO, EnsembleConfigDTO, OptimizationTrialDTO
)
from .explainability_dto import (
    FeatureContributionDTO, LocalExplanationDTO, GlobalExplanationDTO,
    CohortExplanationDTO, ExplanationRequestDTO, MethodComparisonDTO,
    FeatureStatisticsDTO, ExplanationResponseDTO, ExplainInstanceRequestDTO,
    ExplainModelRequestDTO, ExplainCohortRequestDTO, CompareMethodsRequestDTO,
    FeatureRankingDTO, ExplanationSummaryDTO
)

# Backward compatibility aliases
DetectorConfig = CreateDetectorDTO
OptimizationConfig = HyperparameterOptimizationRequestDTO

__all__ = [
    # Detector DTOs
    "DetectorDTO",
    "CreateDetectorDTO", 
    "UpdateDetectorDTO",
    "DetectorResponseDTO",
    "DetectionRequestDTO",
    # Dataset DTOs
    "DatasetDTO",
    "CreateDatasetDTO",
    "DataQualityReportDTO",
    # Result DTOs
    "DetectionResultDTO",
    "AnomalyDTO",
    # Experiment DTOs
    "ExperimentDTO",
    "RunDTO",
    "CreateExperimentDTO",
    "LeaderboardEntryDTO",
    "ExperimentResponseDTO",
    # AutoML DTOs
    "AutoMLRequestDTO",
    "AutoMLResponseDTO", 
    "AutoMLResultDTO",
    "DatasetProfileDTO",
    "AlgorithmRecommendationDTO",
    "HyperparameterOptimizationRequestDTO",
    "HyperparameterOptimizationResponseDTO",
    "AutoMLProfileRequestDTO",
    "AutoMLProfileResponseDTO",
    "EnsembleConfigDTO",
    "OptimizationTrialDTO",
    # Explainability DTOs
    "FeatureContributionDTO",
    "LocalExplanationDTO",
    "GlobalExplanationDTO", 
    "CohortExplanationDTO",
    "ExplanationRequestDTO",
    "MethodComparisonDTO",
    "FeatureStatisticsDTO",
    "ExplanationResponseDTO",
    "ExplainInstanceRequestDTO",
    "ExplainModelRequestDTO",
    "ExplainCohortRequestDTO",
    "CompareMethodsRequestDTO",
    "FeatureRankingDTO",
    "ExplanationSummaryDTO",
    # Backward compatibility aliases
    "DetectorConfig",
    "OptimizationConfig",
]