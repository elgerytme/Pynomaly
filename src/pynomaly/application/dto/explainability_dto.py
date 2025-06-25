"""Data Transfer Objects for explainability operations."""

from __future__ import annotations

from typing import Optional, List, Dict, Any, Union, Literal, Tuple
from pydantic import BaseModel, Field, ConfigDict, validator
from datetime import datetime
from enum import Enum
from uuid import UUID


class FeatureContributionDTO(BaseModel):
    """DTO for feature contribution information."""
    model_config = ConfigDict(from_attributes=True)
    
    feature_name: str = Field(..., description="Name of the feature")
    value: float = Field(..., description="Feature value")
    contribution: float = Field(..., description="Contribution to prediction")
    importance: float = Field(..., description="Absolute importance score")
    rank: int = Field(..., description="Importance rank")
    description: Optional[str] = Field(default=None, description="Description of contribution")


class LocalExplanationDTO(BaseModel):
    """DTO for local explanation."""
    model_config = ConfigDict(from_attributes=True)
    
    instance_id: str = Field(..., description="Instance identifier")
    anomaly_score: float = Field(..., description="Anomaly score")
    prediction: str = Field(..., description="Prediction label")
    confidence: float = Field(..., description="Prediction confidence")
    feature_contributions: List[FeatureContributionDTO] = Field(..., description="Feature contributions")
    explanation_method: str = Field(..., description="Explanation method used")
    model_name: str = Field(..., description="Model name")
    timestamp: str = Field(..., description="Explanation timestamp")


class GlobalExplanationDTO(BaseModel):
    """DTO for global explanation."""
    model_config = ConfigDict(from_attributes=True)
    
    model_name: str = Field(..., description="Model name")
    feature_importances: Dict[str, float] = Field(..., description="Feature importance scores")
    top_features: List[str] = Field(..., description="Top important features")
    explanation_method: str = Field(..., description="Explanation method used")
    model_performance: Dict[str, float] = Field(..., description="Model performance metrics")
    timestamp: str = Field(..., description="Explanation timestamp")
    summary: str = Field(..., description="Explanation summary")


class CohortExplanationDTO(BaseModel):
    """DTO for cohort explanation."""
    model_config = ConfigDict(from_attributes=True)
    
    cohort_id: str = Field(..., description="Cohort identifier")
    cohort_description: str = Field(..., description="Cohort description")
    instance_count: int = Field(..., description="Number of instances in cohort")
    common_features: List[FeatureContributionDTO] = Field(..., description="Common feature contributions")
    explanation_method: str = Field(..., description="Explanation method used")
    model_name: str = Field(..., description="Model name")
    timestamp: str = Field(..., description="Explanation timestamp")


class ExplanationRequestDTO(BaseModel):
    """DTO for explanation request."""
    model_config = ConfigDict(from_attributes=True)
    
    detector_id: str = Field(..., description="Detector identifier")
    dataset_id: Optional[str] = Field(default=None, description="Dataset identifier")
    instance_data: Optional[Dict[str, Any]] = Field(default=None, description="Instance data for explanation")
    instance_indices: Optional[List[int]] = Field(default=None, description="Instance indices to explain")
    explanation_method: str = Field(default="shap", description="Explanation method")
    max_features: int = Field(default=10, ge=1, le=50, description="Maximum features to include")
    background_samples: int = Field(default=100, ge=10, le=1000, description="Background samples for explanation")
    include_cohort_analysis: bool = Field(default=False, description="Include cohort analysis")
    compare_methods: bool = Field(default=False, description="Compare multiple methods")


class MethodComparisonDTO(BaseModel):
    """DTO for method comparison results."""
    model_config = ConfigDict(from_attributes=True)
    
    method_name: str = Field(..., description="Method name")
    success: bool = Field(..., description="Whether explanation succeeded")
    explanation: Optional[Union[LocalExplanationDTO, GlobalExplanationDTO, CohortExplanationDTO]] = Field(
        default=None, description="Generated explanation"
    )
    error: Optional[str] = Field(default=None, description="Error message if failed")
    execution_time: float = Field(..., description="Execution time in seconds")


class FeatureStatisticsDTO(BaseModel):
    """DTO for feature statistics."""
    model_config = ConfigDict(from_attributes=True)
    
    feature_name: str = Field(..., description="Feature name")
    mean_contribution: float = Field(..., description="Mean contribution")
    std_contribution: float = Field(..., description="Standard deviation of contribution")
    mean_importance: float = Field(..., description="Mean importance")
    std_importance: float = Field(..., description="Standard deviation of importance")
    mean_value: float = Field(..., description="Mean feature value")
    std_value: float = Field(..., description="Standard deviation of feature value")
    count: int = Field(..., description="Number of explanations")


class ExplanationResponseDTO(BaseModel):
    """DTO for explanation response."""
    model_config = ConfigDict(from_attributes=True)
    
    success: bool = Field(..., description="Whether explanation succeeded")
    explanations: Optional[Dict[str, Any]] = Field(default=None, description="Generated explanations")
    feature_rankings: Optional[List[tuple]] = Field(default=None, description="Feature importance rankings")
    cohort_analysis: Optional[Dict[str, Any]] = Field(default=None, description="Cohort analysis results")
    method_comparison: Optional[Dict[str, MethodComparisonDTO]] = Field(default=None, description="Method comparison results")
    feature_statistics: Optional[Dict[str, FeatureStatisticsDTO]] = Field(default=None, description="Feature statistics")
    available_methods: Optional[List[str]] = Field(default=None, description="Available explanation methods")
    message: str = Field(..., description="Response message")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    execution_time: float = Field(..., description="Total execution time")


class ExplainInstanceRequestDTO(BaseModel):
    """DTO for single instance explanation request."""
    model_config = ConfigDict(from_attributes=True)
    
    detector_id: str = Field(..., description="Detector identifier")
    instance_data: Dict[str, Any] = Field(..., description="Instance data to explain")
    explanation_method: str = Field(default="shap", description="Explanation method")
    max_features: int = Field(default=10, ge=1, le=50, description="Maximum features to show")


class ExplainModelRequestDTO(BaseModel):
    """DTO for global model explanation request."""
    model_config = ConfigDict(from_attributes=True)
    
    detector_id: str = Field(..., description="Detector identifier")
    dataset_id: str = Field(..., description="Dataset identifier for background data")
    explanation_method: str = Field(default="shap", description="Explanation method")
    max_features: int = Field(default=10, ge=1, le=50, description="Maximum features to show")
    background_samples: int = Field(default=100, ge=10, le=1000, description="Background samples")


class ExplainCohortRequestDTO(BaseModel):
    """DTO for cohort explanation request."""
    model_config = ConfigDict(from_attributes=True)
    
    detector_id: str = Field(..., description="Detector identifier")
    dataset_id: str = Field(..., description="Dataset identifier")
    instance_indices: List[int] = Field(..., description="Indices of instances in cohort")
    explanation_method: str = Field(default="shap", description="Explanation method")
    max_features: int = Field(default=10, ge=1, le=50, description="Maximum features to show")
    cohort_name: Optional[str] = Field(default=None, description="Optional cohort name")


class CompareMethodsRequestDTO(BaseModel):
    """DTO for comparing explanation methods."""
    model_config = ConfigDict(from_attributes=True)
    
    detector_id: str = Field(..., description="Detector identifier")
    instance_data: Optional[Dict[str, Any]] = Field(default=None, description="Instance data")
    dataset_id: Optional[str] = Field(default=None, description="Dataset identifier")
    instance_index: Optional[int] = Field(default=None, description="Instance index if using dataset")
    methods: List[str] = Field(..., description="Methods to compare")
    max_features: int = Field(default=10, ge=1, le=50, description="Maximum features to show")


class FeatureRankingDTO(BaseModel):
    """DTO for feature ranking information."""
    model_config = ConfigDict(from_attributes=True)
    
    feature_name: str = Field(..., description="Feature name")
    importance_score: float = Field(..., description="Average importance score")
    rank: int = Field(..., description="Feature rank")
    frequency: int = Field(..., description="Frequency in top features")
    variance: float = Field(..., description="Importance variance across explanations")


class ExplanationSummaryDTO(BaseModel):
    """DTO for explanation summary."""
    model_config = ConfigDict(from_attributes=True)
    
    detector_id: str = Field(..., description="Detector identifier")
    dataset_id: Optional[str] = Field(default=None, description="Dataset identifier")
    total_explanations: int = Field(..., description="Total explanations generated")
    methods_used: List[str] = Field(..., description="Methods used")
    top_features: List[FeatureRankingDTO] = Field(..., description="Top feature rankings")
    execution_summary: Dict[str, float] = Field(..., description="Execution time summary")
    created_at: datetime = Field(default_factory=datetime.now, description="Summary creation time")