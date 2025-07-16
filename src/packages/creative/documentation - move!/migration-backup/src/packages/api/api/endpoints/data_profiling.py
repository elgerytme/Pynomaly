"""Data Profiling API Endpoints.

This module provides RESTful endpoints for data profiling operations including
dataset profiling, schema analysis, and statistical analysis.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from ...data_profiling.application.services.profiling_engine import (
    ProfilingEngine, ProfilingConfig)
from ...data_profiling.domain.entities.data_profile import DatasetId
from ..security.authorization import require_permissions
from ..dependencies.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/data-profiling", tags=["Data Profiling"])

# Pydantic models for request/response
class DataProfilingRequest(BaseModel):
    """Request model for data profiling."""
    dataset_id: str = Field(..., description="Unique identifier for the dataset")
    data: List[Dict[str, Any]] = Field(..., description="Dataset records as list of dictionaries")
    config: Optional[Dict[str, Any]] = Field(default=None, description="Profiling configuration options")        schema_extra = {
            "example": {
                "dataset_id": "customer_data_2024",
                "data": [
                    {"id": 1, "name": "John Doe", "age": 30, "email": "john@example.com"},
                    {"id": 2, "name": "Jane Smith", "age": 25, "email": "jane@example.com"}
                ],
                "config": {
                    "enable_sampling": True,
                    "sample_size": 1000,
                    "enable_parallel_processing": True
                }
            }
        }


class DataProfilingResponse(BaseModel):
    """Response model for data profiling."""
    profile_id: str = Field(..., description="Unique identifier for the profile")
    dataset_id: str = Field(..., description="Dataset identifier")
    overall_score: float = Field(..., description="Overall quality score")
    quality_scores: Dict[str, float] = Field(..., description="Quality scores by dimension")
    total_records: int = Field(..., description="Total number of records")
    total_columns: int = Field(..., description="Total number of columns")
    issues_detected: int = Field(..., description="Number of quality issues detected")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    created_at: str = Field(..., description="Profile creation timestamp")        schema_extra = {
            "example": {
                "profile_id": "prof_123456789",
                "dataset_id": "customer_data_2024",
                "overall_score": 0.85,
                "quality_scores": {
                    "completeness": 0.92,
                    "accuracy": 0.88,
                    "consistency": 0.90,
                    "validity": 0.82,
                    "uniqueness": 0.95,
                    "timeliness": 0.78
                },
                "total_records": 1000,
                "total_columns": 5,
                "issues_detected": 12,
                "processing_time_ms": 245.7,
                "created_at": "2024-01-15T10:30:00Z"
            }
        }


class ProfileListResponse(BaseModel):
    """Response model for profile listing."""
    profiles: List[Dict[str, Any]] = Field(..., description="List of data profiles")
    total_count: int = Field(..., description="Total number of profiles")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Number of profiles per page")


class DetailedProfileResponse(BaseModel):
    """Response model for detailed profile information."""
    profile: Dict[str, Any] = Field(..., description="Complete profile information")
    quality_issues: List[Dict[str, Any]] = Field(..., description="List of quality issues")
    remediation_suggestions: List[Dict[str, Any]] = Field(..., description="Remediation suggestions")
    quality_trends: Dict[str, Any] = Field(..., description="Quality trend analysis")


# Initialize profiling engine
profiling_engine = ProfilingEngine()


@router.post(
    "/profile",
    response_model=DataProfilingResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Profile dataset",
    description="Perform comprehensive data profiling on the provided dataset"
)
async def profile_dataset(
    request: DataProfilingRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    _: None = Depends(require_permissions(["data_profiling:write"]))
) -> DataProfilingResponse:
    """Profile a dataset and return comprehensive quality analysis."""
    try:
        start_time = datetime.now()
        
        # Convert request data to DataFrame
        df = pd.DataFrame(request.data)
        
        if df.empty:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Dataset cannot be empty"
            )
        
        # Configure profiling engine
        config = ProfilingConfig()
        if request.config:
            config = ProfilingConfig(**request.config)
        
        # Update engine configuration
        profiling_engine.config = config
        
        # Create dataset ID
        dataset_id = DatasetId(request.dataset_id)
        
        # Execute profiling
        profile = profiling_engine.profile_dataset(df, dataset_id=dataset_id)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Build response
        response = DataProfilingResponse(
            profile_id=str(profile.profile_id),
            dataset_id=str(profile.dataset_id),
            overall_score=profile.quality_scores.overall_score,
            quality_scores=profile.quality_scores.get_dimension_scores(),
            total_records=profile.record_count or len(df),
            total_columns=profile.column_count or len(df.columns),
            issues_detected=len(profile.quality_issues),
            processing_time_ms=processing_time,
            created_at=profile.created_at.isoformat()
        )
        
        logger.info(f"Dataset profiled successfully: {request.dataset_id} by user {current_user['user_id']}")
        return response
        
    except ValueError as e:
        logger.error(f"Invalid data profiling request: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Data profiling failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Data profiling operation failed"
        )


@router.get(
    "/profiles",
    response_model=ProfileListResponse,
    summary="List data profiles",
    description="Get a paginated list of data profiles"
)
async def list_profiles(
    page: int = 1,
    page_size: int = 20,
    dataset_id: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_user),
    _: None = Depends(require_permissions(["data_profiling:read"]))
) -> ProfileListResponse:
    """Get a list of data profiles with optional filtering."""
    try:
        # This would typically query a database
        # For demonstration, return mock data
        
        mock_profiles = [
            {
                "profile_id": f"prof_{i:06d}",
                "dataset_id": f"dataset_{i}",
                "overall_score": 0.75 + (i % 25) / 100,
                "total_records": 1000 + i * 100,
                "total_columns": 5 + i % 10,
                "issues_detected": i % 15,
                "created_at": f"2024-01-{(i % 28) + 1:02d}T10:00:00Z"
            }
            for i in range(1, 101)  # Mock 100 profiles
        ]
        
        # Apply filtering
        if dataset_id:
            mock_profiles = [p for p in mock_profiles if p["dataset_id"] == dataset_id]
        
        # Apply pagination
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_profiles = mock_profiles[start_idx:end_idx]
        
        return ProfileListResponse(
            profiles=paginated_profiles,
            total_count=len(mock_profiles),
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        logger.error(f"Failed to list profiles: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve profiles"
        )


@router.get(
    "/profiles/{profile_id}",
    response_model=DetailedProfileResponse,
    summary="Get detailed profile",
    description="Get detailed information about a specific data profile"
)
async def get_profile_details(
    profile_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    _: None = Depends(require_permissions(["data_profiling:read"]))
) -> DetailedProfileResponse:
    """Get detailed information about a specific data profile."""
    try:
        # This would typically query a database
        # For demonstration, return mock detailed data
        
        mock_profile = {
            "profile_id": profile_id,
            "dataset_id": "customer_data_2024",
            "overall_score": 0.85,
            "quality_scores": {
                "completeness": 0.92,
                "accuracy": 0.88,
                "consistency": 0.90,
                "validity": 0.82,
                "uniqueness": 0.95,
                "timeliness": 0.78
            },
            "total_records": 10000,
            "total_columns": 12,
            "data_size_bytes": 2048000,
            "version": "1.0.0",
            "created_at": "2024-01-15T10:30:00Z",
            "last_assessed": "2024-01-15T10:30:00Z"
        }
        
        mock_issues = [
            {
                "issue_id": "issue_001",
                "issue_type": "missing_values",
                "severity": "medium",
                "description": "Email column has 15% missing values",
                "affected_records": 1500,
                "affected_columns": ["email"],
                "detected_at": "2024-01-15T10:30:00Z"
            },
            {
                "issue_id": "issue_002",
                "issue_type": "invalid_format",
                "severity": "high",
                "description": "Phone numbers have inconsistent formats",
                "affected_records": 250,
                "affected_columns": ["phone"],
                "detected_at": "2024-01-15T10:30:00Z"
            }
        ]
        
        mock_suggestions = [
            {
                "suggestion_id": "sugg_001",
                "issue_id": "issue_001",
                "action_type": "data_cleansing",
                "description": "Implement email validation at data entry points",
                "effort_estimate": "moderate",
                "success_probability": 0.85,
                "priority": "medium"
            },
            {
                "suggestion_id": "sugg_002",
                "issue_id": "issue_002",
                "action_type": "data_cleansing",
                "description": "Standardize phone number formats using regex transformation",
                "effort_estimate": "minor",
                "success_probability": 0.95,
                "priority": "high"
            }
        ]
        
        mock_trends = {
            "trend_direction": "stable",
            "trend_strength": 0.02,
            "average_score": 0.84,
            "score_variance": 0.001,
            "data_points": 30,
            "analysis_date": "2024-01-15T10:30:00Z"
        }
        
        return DetailedProfileResponse(
            profile=mock_profile,
            quality_issues=mock_issues,
            remediation_suggestions=mock_suggestions,
            quality_trends=mock_trends
        )
        
    except Exception as e:
        logger.error(f"Failed to get profile details: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Profile not found"
        )


@router.delete(
    "/profiles/{profile_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete profile",
    description="Delete a data profile"
)
async def delete_profile(
    profile_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    _: None = Depends(require_permissions(["data_profiling:delete"]))
) -> None:
    """Delete a data profile."""
    try:
        # This would typically delete from database
        # For demonstration, just log the operation
        
        logger.info(f"Profile {profile_id} deleted by user {current_user['user_id']}")
        
    except Exception as e:
        logger.error(f"Failed to delete profile: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete profile"
        )


@router.post(
    "/profiles/{profile_id}/compare/{other_profile_id}",
    summary="Compare profiles",
    description="Compare two data profiles to identify changes and trends"
)
async def compare_profiles(
    profile_id: str,
    other_profile_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    _: None = Depends(require_permissions(["data_profiling:read"]))
) -> Dict[str, Any]:
    """Compare two data profiles."""
    try:
        # This would typically load profiles from database and compare them
        # For demonstration, return mock comparison data
        
        comparison_result = {
            "profile1_id": profile_id,
            "profile2_id": other_profile_id,
            "comparison_date": datetime.now().isoformat(),
            "score_comparison": {
                "overall_delta": 0.05,
                "completeness_delta": 0.02,
                "accuracy_delta": 0.03,
                "consistency_delta": -0.01,
                "validity_delta": 0.04,
                "uniqueness_delta": 0.00,
                "timeliness_delta": 0.02
            },
            "issue_comparison": {
                "total_issues_1": 15,
                "total_issues_2": 12,
                "issue_change": -3,
                "new_issue_types": [],
                "resolved_issue_types": ["duplicate_records"]
            },
            "recommendations": [
                "Quality has improved slightly - continue current practices",
                "Address remaining accuracy issues for further improvement"
            ]
        }
        
        logger.info(f"Profiles compared: {profile_id} vs {other_profile_id} by user {current_user['user_id']}")
        return comparison_result
        
    except Exception as e:
        logger.error(f"Failed to compare profiles: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to compare profiles"
        )


@router.get(
    "/engine/metrics",
    summary="Get engine metrics",
    description="Get performance metrics for the profiling engine"
)
async def get_engine_metrics(
    current_user: Dict[str, Any] = Depends(get_current_user),
    _: None = Depends(require_permissions(["data_profiling:read"]))
) -> Dict[str, Any]:
    """Get profiling engine performance metrics."""
    try:
        metrics = profiling_engine.get_execution_metrics()
        cache_info = profiling_engine.get_cache_info()
        
        return {
            "execution_metrics": metrics,
            "cache_info": cache_info,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get engine metrics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve engine metrics"
        )


@router.post(
    "/engine/cache/clear",
    summary="Clear engine cache",
    description="Clear the profiling engine cache"
)
async def clear_engine_cache(
    current_user: Dict[str, Any] = Depends(get_current_user),
    _: None = Depends(require_permissions(["data_profiling:admin"]))
) -> Dict[str, str]:
    """Clear the profiling engine cache."""
    try:
        profiling_engine.clear_cache()
        
        logger.info(f"Profiling engine cache cleared by user {current_user['user_id']}")
        return {"message": "Cache cleared successfully"}
        
    except Exception as e:
        logger.error(f"Failed to clear cache: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear cache"
        )