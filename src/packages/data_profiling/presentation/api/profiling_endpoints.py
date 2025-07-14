"""Data Profiling API endpoints."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

import pandas as pd
import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ...application.use_cases.execute_data_profiling import ExecuteDataProfilingUseCase
from ...domain.entities.data_profile import (
    DataProfile, ProfileId, DatasetId, ProfilingStatus,
    ColumnProfile, QualityIssue, Pattern
)
from ...infrastructure.adapters.in_memory_data_profile_repository import (
    InMemoryDataProfileRepository
)

logger = structlog.get_logger(__name__)

# Request Models
class CreateProfilingRequest(BaseModel):
    """Request model for creating data profiling job."""
    dataset_id: str = Field(..., description="ID of the dataset to profile")
    source_type: str = Field(default="dataframe", description="Type of data source")
    source_connection: Dict[str, Any] = Field(
        default_factory=dict,
        description="Connection details for data source"
    )
    profiling_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Profiling configuration parameters"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "dataset_id": "550e8400-e29b-41d4-a716-446655440000",
                "source_type": "database",
                "source_connection": {
                    "host": "localhost",
                    "database": "analytics",
                    "table": "customers"
                },
                "profiling_config": {
                    "strategy": "sample",
                    "sample_percentage": 10.0,
                    "include_patterns": True,
                    "include_statistics": True
                }
            }
        }

class ProfilingDataRequest(BaseModel):
    """Request model for providing data for profiling."""
    data: List[Dict[str, Any]] = Field(..., description="Dataset records")
    
    class Config:
        json_schema_extra = {
            "example": {
                "data": [
                    {"id": 1, "name": "John Doe", "email": "john@example.com", "age": 30},
                    {"id": 2, "name": "Jane Smith", "email": "jane@example.com", "age": 25},
                    {"id": 3, "name": "Bob Johnson", "email": "bob@example.com", "age": 35}
                ]
            }
        }

# Response Models
class PatternResponse(BaseModel):
    """Response model for detected patterns."""
    pattern_type: str
    regex: str
    frequency: int
    percentage: float
    examples: List[str]
    confidence: float

class QualityIssueResponse(BaseModel):
    """Response model for quality issues."""
    issue_type: str
    severity: str
    description: str
    affected_rows: int
    affected_percentage: float
    examples: List[str]
    suggested_action: Optional[str]

class ValueDistributionResponse(BaseModel):
    """Response model for value distribution."""
    unique_count: int
    null_count: int
    total_count: int
    completeness_ratio: float
    top_values: Dict[str, int]

class StatisticalSummaryResponse(BaseModel):
    """Response model for statistical summary."""
    min_value: Optional[float]
    max_value: Optional[float]
    mean: Optional[float]
    median: Optional[float]
    std_dev: Optional[float]
    quartiles: Optional[List[float]]

class ColumnProfileResponse(BaseModel):
    """Response model for column profile."""
    column_name: str
    data_type: str
    inferred_type: Optional[str]
    nullable: bool
    distribution: ValueDistributionResponse
    cardinality: str
    statistical_summary: Optional[StatisticalSummaryResponse]
    patterns: List[PatternResponse]
    quality_score: float
    quality_issues: List[QualityIssueResponse]
    semantic_type: Optional[str]
    business_meaning: Optional[str]

class SchemaProfileResponse(BaseModel):
    """Response model for schema profile."""
    table_name: str
    total_columns: int
    total_rows: int
    columns: List[ColumnProfileResponse]
    primary_keys: List[str]
    foreign_keys: Dict[str, str]
    unique_constraints: List[List[str]]
    check_constraints: List[str]
    estimated_size_bytes: Optional[int]
    compression_ratio: Optional[float]

class QualityAssessmentResponse(BaseModel):
    """Response model for quality assessment."""
    overall_score: float
    completeness_score: float
    consistency_score: float
    accuracy_score: float
    validity_score: float
    uniqueness_score: float
    dimension_weights: Dict[str, float]
    critical_issues: int
    high_issues: int
    medium_issues: int
    low_issues: int
    recommendations: List[str]

class ProfilingMetadataResponse(BaseModel):
    """Response model for profiling metadata."""
    profiling_strategy: str
    sample_size: Optional[int]
    sample_percentage: Optional[float]
    execution_time_seconds: float
    memory_usage_mb: Optional[float]
    include_patterns: bool
    include_statistical_analysis: bool
    include_quality_assessment: bool

class ProfileResponse(BaseModel):
    """Response model for data profile."""
    profile_id: str
    dataset_id: str
    status: str
    schema_profile: Optional[SchemaProfileResponse]
    quality_assessment: Optional[QualityAssessmentResponse]
    profiling_metadata: Optional[ProfilingMetadataResponse]
    started_at: Optional[str]
    completed_at: Optional[str]
    error_message: Optional[str]
    source_type: str
    source_connection: Dict[str, Any]
    source_query: Optional[str]
    created_at: str

class ProfileListResponse(BaseModel):
    """Response model for profile list."""
    profiles: List[ProfileResponse]
    total_count: int
    page: int
    page_size: int

class TaskResponse(BaseModel):
    """Response model for background tasks."""
    task_id: str
    status: str
    message: str
    profile_id: str

# Router setup
router = APIRouter(
    prefix="/data-profiling",
    tags=["Data Profiling"],
    responses={
        400: {"description": "Bad Request"},
        404: {"description": "Not Found"},
        422: {"description": "Validation Error"},
        500: {"description": "Internal Server Error"},
    },
)

# Dependency injection setup
def get_repository():
    """Get repository instance."""
    return InMemoryDataProfileRepository()

def get_use_case(repository = Depends(get_repository)):
    """Get use case instance."""
    return ExecuteDataProfilingUseCase(repository)

# Background task functions
async def execute_profiling_background(
    profile_id: str,
    data_records: List[Dict[str, Any]],
    profiling_config: Dict[str, Any],
    use_case: ExecuteDataProfilingUseCase
):
    """Execute profiling in background."""
    try:
        # Convert data to DataFrame
        data = pd.DataFrame(data_records)
        
        # Get profile entity
        profile = await use_case.get_profile_by_id(
            ProfileId(value=UUID(profile_id))
        )
        
        if profile and profile.status == ProfilingStatus.PENDING:
            # Execute profiling (would call actual profiling service)
            # For now, using the fallback implementation
            schema_profile = use_case._create_basic_schema_profile(data)
            quality_assessment = use_case._create_basic_quality_assessment(data)
            metadata = use_case._create_basic_metadata(data)
            
            profile.complete_profiling(
                schema_profile=schema_profile,
                quality_assessment=quality_assessment,
                metadata=metadata
            )
            
            await use_case.repository.save(profile)
            
        logger.info("Background profiling completed", profile_id=profile_id)
        
    except Exception as e:
        logger.error("Background profiling failed", profile_id=profile_id, error=str(e))

# Endpoints
@router.post(
    "/",
    response_model=TaskResponse,
    summary="Create Data Profiling Job",
    description="""
    Create a new data profiling job to analyze dataset characteristics.
    
    **Features:**
    - Automatic schema discovery and data type inference
    - Value distribution analysis and cardinality assessment
    - Pattern recognition (emails, phones, URLs, dates)
    - Statistical profiling for numerical data
    - Data quality assessment across multiple dimensions
    - Completeness, consistency, and validity analysis
    
    **Profiling Strategies:**
    - full: Complete dataset analysis
    - sample: Sample-based profiling for large datasets
    - incremental: Update existing profiles with new data
    """,
    responses={
        200: {"description": "Profiling job created successfully"},
        400: {"description": "Invalid request parameters"},
        500: {"description": "Profiling job creation failed"}
    }
)
async def create_profiling(
    request: CreateProfilingRequest,
    data_request: ProfilingDataRequest,
    background_tasks: BackgroundTasks,
    use_case: ExecuteDataProfilingUseCase = Depends(get_use_case)
):
    """Create a new data profiling job."""
    try:
        # Convert data to DataFrame for validation
        data = pd.DataFrame(data_request.data)
        
        if data.empty:
            raise HTTPException(
                status_code=400,
                detail="Dataset cannot be empty"
            )
        
        # Execute profiling
        profile = await use_case.execute(
            dataset_id=DatasetId(value=UUID(request.dataset_id)),
            data=data,
            source_type=request.source_type,
            source_connection=request.source_connection,
            profiling_config=request.profiling_config
        )
        
        # Add background task for detailed processing
        background_tasks.add_task(
            execute_profiling_background,
            str(profile.profile_id.value),
            data_request.data,
            request.profiling_config,
            use_case
        )
        
        return TaskResponse(
            task_id=str(profile.profile_id.value),
            status="pending",
            message="Data profiling submitted for processing",
            profile_id=str(profile.profile_id.value)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to create profiling job", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Profiling job creation failed: {str(e)}"
        )

def _convert_profile_to_response(profile: DataProfile) -> ProfileResponse:
    """Convert profile entity to response model."""
    schema_response = None
    if profile.schema_profile:
        column_responses = []
        for col in profile.schema_profile.columns:
            # Convert patterns
            pattern_responses = [
                PatternResponse(
                    pattern_type=p.pattern_type.value,
                    regex=p.regex,
                    frequency=p.frequency,
                    percentage=p.percentage,
                    examples=p.examples,
                    confidence=p.confidence
                )
                for p in col.patterns
            ]
            
            # Convert quality issues
            issue_responses = [
                QualityIssueResponse(
                    issue_type=issue.issue_type.value,
                    severity=issue.severity,
                    description=issue.description,
                    affected_rows=issue.affected_rows,
                    affected_percentage=issue.affected_percentage,
                    examples=issue.examples,
                    suggested_action=issue.suggested_action
                )
                for issue in col.quality_issues
            ]
            
            # Convert statistical summary
            stats_response = None
            if col.statistical_summary:
                stats_response = StatisticalSummaryResponse(
                    min_value=col.statistical_summary.min_value,
                    max_value=col.statistical_summary.max_value,
                    mean=col.statistical_summary.mean,
                    median=col.statistical_summary.median,
                    std_dev=col.statistical_summary.std_dev,
                    quartiles=col.statistical_summary.quartiles
                )
            
            column_responses.append(ColumnProfileResponse(
                column_name=col.column_name,
                data_type=col.data_type.value,
                inferred_type=col.inferred_type.value if col.inferred_type else None,
                nullable=col.nullable,
                distribution=ValueDistributionResponse(
                    unique_count=col.distribution.unique_count,
                    null_count=col.distribution.null_count,
                    total_count=col.distribution.total_count,
                    completeness_ratio=col.distribution.completeness_ratio,
                    top_values=col.distribution.top_values
                ),
                cardinality=col.cardinality.value,
                statistical_summary=stats_response,
                patterns=pattern_responses,
                quality_score=col.quality_score,
                quality_issues=issue_responses,
                semantic_type=col.semantic_type,
                business_meaning=col.business_meaning
            ))
        
        schema_response = SchemaProfileResponse(
            table_name=profile.schema_profile.table_name,
            total_columns=profile.schema_profile.total_columns,
            total_rows=profile.schema_profile.total_rows,
            columns=column_responses,
            primary_keys=profile.schema_profile.primary_keys,
            foreign_keys=profile.schema_profile.foreign_keys,
            unique_constraints=profile.schema_profile.unique_constraints,
            check_constraints=profile.schema_profile.check_constraints,
            estimated_size_bytes=profile.schema_profile.estimated_size_bytes,
            compression_ratio=profile.schema_profile.compression_ratio
        )
    
    # Convert quality assessment
    quality_response = None
    if profile.quality_assessment:
        quality_response = QualityAssessmentResponse(
            overall_score=profile.quality_assessment.overall_score,
            completeness_score=profile.quality_assessment.completeness_score,
            consistency_score=profile.quality_assessment.consistency_score,
            accuracy_score=profile.quality_assessment.accuracy_score,
            validity_score=profile.quality_assessment.validity_score,
            uniqueness_score=profile.quality_assessment.uniqueness_score,
            dimension_weights=profile.quality_assessment.dimension_weights,
            critical_issues=profile.quality_assessment.critical_issues,
            high_issues=profile.quality_assessment.high_issues,
            medium_issues=profile.quality_assessment.medium_issues,
            low_issues=profile.quality_assessment.low_issues,
            recommendations=profile.quality_assessment.recommendations
        )
    
    # Convert metadata
    metadata_response = None
    if profile.profiling_metadata:
        metadata_response = ProfilingMetadataResponse(
            profiling_strategy=profile.profiling_metadata.profiling_strategy,
            sample_size=profile.profiling_metadata.sample_size,
            sample_percentage=profile.profiling_metadata.sample_percentage,
            execution_time_seconds=profile.profiling_metadata.execution_time_seconds,
            memory_usage_mb=profile.profiling_metadata.memory_usage_mb,
            include_patterns=profile.profiling_metadata.include_patterns,
            include_statistical_analysis=profile.profiling_metadata.include_statistical_analysis,
            include_quality_assessment=profile.profiling_metadata.include_quality_assessment
        )
    
    return ProfileResponse(
        profile_id=str(profile.profile_id.value),
        dataset_id=str(profile.dataset_id.value),
        status=profile.status.value,
        schema_profile=schema_response,
        quality_assessment=quality_response,
        profiling_metadata=metadata_response,
        started_at=profile.started_at.isoformat() if profile.started_at else None,
        completed_at=profile.completed_at.isoformat() if profile.completed_at else None,
        error_message=profile.error_message,
        source_type=profile.source_type,
        source_connection=profile.source_connection,
        source_query=profile.source_query,
        created_at=profile.created_at.isoformat()
    )

@router.get(
    "/{profile_id}",
    response_model=ProfileResponse,
    summary="Get Data Profile",
    description="Retrieve a specific data profile by ID.",
    responses={
        200: {"description": "Profile retrieved successfully"},
        404: {"description": "Profile not found"}
    }
)
async def get_profile(
    profile_id: str,
    use_case: ExecuteDataProfilingUseCase = Depends(get_use_case)
):
    """Get profile by ID."""
    try:
        profile = await use_case.get_profile_by_id(
            ProfileId(value=UUID(profile_id))
        )
        
        if not profile:
            raise HTTPException(status_code=404, detail="Profile not found")
        
        return _convert_profile_to_response(profile)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get profile", profile_id=profile_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve profile: {str(e)}"
        )

@router.get(
    "/",
    response_model=ProfileListResponse,
    summary="List Data Profiles",
    description="List data profiles with optional filtering.",
    responses={
        200: {"description": "Profiles retrieved successfully"}
    }
)
async def list_profiles(
    dataset_id: Optional[str] = Query(None, description="Filter by dataset ID"),
    status: Optional[str] = Query(None, description="Filter by status"),
    source_type: Optional[str] = Query(None, description="Filter by source type"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Items per page"),
    use_case: ExecuteDataProfilingUseCase = Depends(get_use_case)
):
    """List profiles with filtering and pagination."""
    try:
        # Get profiles based on filters
        if dataset_id:
            profiles = await use_case.get_profiles_by_dataset(
                DatasetId(value=UUID(dataset_id))
            )
        else:
            profiles = await use_case.repository.list_all()
        
        # Apply additional filters
        if status:
            try:
                status_enum = ProfilingStatus(status)
                profiles = [p for p in profiles if p.status == status_enum]
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
        
        if source_type:
            profiles = [p for p in profiles if p.source_type == source_type]
        
        # Calculate pagination
        total_count = len(profiles)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_profiles = profiles[start_idx:end_idx]
        
        # Convert to response models
        profile_responses = [
            _convert_profile_to_response(profile)
            for profile in paginated_profiles
        ]
        
        return ProfileListResponse(
            profiles=profile_responses,
            total_count=total_count,
            page=page,
            page_size=page_size
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to list profiles", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve profiles: {str(e)}"
        )

@router.get(
    "/dataset/{dataset_id}/latest",
    response_model=ProfileResponse,
    summary="Get Latest Profile for Dataset",
    description="Get the most recent profile for a specific dataset.",
    responses={
        200: {"description": "Latest profile retrieved successfully"},
        404: {"description": "No profiles found for dataset"}
    }
)
async def get_latest_profile(
    dataset_id: str,
    use_case: ExecuteDataProfilingUseCase = Depends(get_use_case)
):
    """Get latest profile for a dataset."""
    try:
        profile = await use_case.get_latest_profile_by_dataset(
            DatasetId(value=UUID(dataset_id))
        )
        
        if not profile:
            raise HTTPException(
                status_code=404, 
                detail="No profiles found for this dataset"
            )
        
        return _convert_profile_to_response(profile)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get latest profile", dataset_id=dataset_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve latest profile: {str(e)}"
        )

@router.get(
    "/health",
    summary="Health Check",
    description="Check the health of the data profiling service.",
    responses={
        200: {"description": "Service is healthy"},
        503: {"description": "Service is unavailable"}
    }
)
async def health_check():
    """Health check endpoint."""
    try:
        return {
            "status": "healthy",
            "service": "data-profiling",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0"
        }
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(
            status_code=503,
            detail="Service unavailable"
        )