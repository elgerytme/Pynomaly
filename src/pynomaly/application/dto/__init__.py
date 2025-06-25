"""Data Transfer Objects for application layer."""

from .automl_dto import (
    AlgorithmRecommendationDTO,
    AutoMLProfileRequestDTO,
    AutoMLProfileResponseDTO,
    AutoMLRequestDTO,
    AutoMLResponseDTO,
    AutoMLResultDTO,
    DatasetProfileDTO,
    EnsembleConfigDTO,
    HyperparameterOptimizationRequestDTO,
    HyperparameterOptimizationResponseDTO,
    OptimizationTrialDTO,
)
from .dataset_dto import CreateDatasetDTO, DataQualityReportDTO, DatasetDTO
from .detection_dto import (
    DetectionRequestDTO,
    DetectionSummaryDTO,
    ExplanationResultDTO,
    TrainingRequestDTO,
    TrainingResultDTO,
)
from .detector_dto import (
    CreateDetectorDTO,
    DetectorDTO,
    DetectorResponseDTO,
    UpdateDetectorDTO,
)
from .experiment_dto import (
    CreateExperimentDTO,
    ExperimentDTO,
    ExperimentResponseDTO,
    LeaderboardEntryDTO,
    RunDTO,
)
from .explainability_dto import (
    CohortExplanationDTO,
    CompareMethodsRequestDTO,
    ExplainCohortRequestDTO,
    ExplainInstanceRequestDTO,
    ExplainModelRequestDTO,
    ExplanationRequestDTO,
    ExplanationResponseDTO,
    ExplanationSummaryDTO,
    FeatureContributionDTO,
    FeatureRankingDTO,
    FeatureStatisticsDTO,
    GlobalExplanationDTO,
    LocalExplanationDTO,
    MethodComparisonDTO,
)
from .result_dto import AnomalyDTO, DetectionResultDTO

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
    "DetectionSummaryDTO",
    "ExplanationResultDTO",
    "TrainingRequestDTO",
    "TrainingResultDTO",
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
