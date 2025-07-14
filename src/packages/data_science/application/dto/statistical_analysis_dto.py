"""Data Transfer Objects for statistical analysis operations."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from uuid import UUID
from datetime import datetime

from pydantic import BaseModel, Field


@dataclass(frozen=True)
class StatisticalAnalysisRequestDTO:
    """Request DTO for statistical analysis."""
    dataset_id: UUID
    user_id: UUID
    analysis_type: str
    target_column: Optional[str] = None
    feature_columns: Optional[List[str]] = None
    analysis_params: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class StatisticalAnalysisResponseDTO:
    """Response DTO for statistical analysis."""
    analysis_id: UUID
    status: str
    results: Optional[Dict[str, Any]] = None
    insights: Optional[List[str]] = None
    error_message: Optional[str] = None
    execution_time_seconds: Optional[float] = None
    created_at: datetime


@dataclass(frozen=True)
class CorrelationAnalysisRequestDTO:
    """Request DTO for correlation analysis."""
    dataset_id: UUID
    user_id: UUID
    features: Optional[List[str]] = None
    correlation_method: str = "pearson"
    significance_level: float = 0.05
    include_p_values: bool = True


@dataclass(frozen=True)
class CorrelationAnalysisResponseDTO:
    """Response DTO for correlation analysis."""
    analysis_id: UUID
    correlation_matrix: List[List[float]]
    feature_names: List[str]
    p_value_matrix: Optional[List[List[float]]] = None
    significant_correlations: Optional[List[Dict[str, Any]]] = None
    multicollinearity_warnings: Optional[List[str]] = None


@dataclass(frozen=True)
class DistributionAnalysisRequestDTO:
    """Request DTO for distribution analysis."""
    dataset_id: UUID
    user_id: UUID
    feature: str
    distribution_tests: Optional[List[str]] = None
    confidence_level: float = 0.95


@dataclass(frozen=True)
class DistributionAnalysisResponseDTO:
    """Response DTO for distribution analysis."""
    analysis_id: UUID
    feature: str
    distribution_type: str
    distribution_parameters: Dict[str, float]
    goodness_of_fit_tests: List[Dict[str, Any]]
    normality_test_results: Dict[str, Any]
    outlier_analysis: Dict[str, Any]