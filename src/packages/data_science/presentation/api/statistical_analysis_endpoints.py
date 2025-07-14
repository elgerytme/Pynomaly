"""Statistical Analysis API endpoints."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

import pandas as pd
import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ...application.use_cases.execute_statistical_analysis import ExecuteStatisticalAnalysisUseCase
from ...domain.entities.statistical_analysis import (
    StatisticalAnalysis, StatisticalAnalysisId, DatasetId, UserId, AnalysisType
)
from ...infrastructure.adapters.in_memory_statistical_analysis_repository import (
    InMemoryStatisticalAnalysisRepository
)

logger = structlog.get_logger(__name__)

# Request Models
class CreateAnalysisRequest(BaseModel):
    """Request model for creating statistical analysis."""
    dataset_id: str = Field(..., description="ID of the dataset to analyze")
    analysis_type: str = Field(..., description="Type of analysis to perform")
    feature_columns: List[str] = Field(..., description="Columns to include in analysis")
    target_column: Optional[str] = Field(None, description="Target column for supervised analysis")
    analysis_params: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Additional analysis parameters"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "dataset_id": "550e8400-e29b-41d4-a716-446655440000",
                "analysis_type": "descriptive_statistics",
                "feature_columns": ["age", "income", "score"],
                "target_column": "target",
                "analysis_params": {
                    "confidence_level": 0.95,
                    "hypothesis_tests": ["t_test", "normality_test"]
                }
            }
        }

class AnalysisDataRequest(BaseModel):
    """Request model for providing data for analysis."""
    data: List[Dict[str, Any]] = Field(..., description="Dataset records")
    
    class Config:
        json_schema_extra = {
            "example": {
                "data": [
                    {"age": 25, "income": 50000, "score": 85},
                    {"age": 30, "income": 60000, "score": 90},
                    {"age": 35, "income": 70000, "score": 88}
                ]
            }
        }

# Response Models
class StatisticalTestResponse(BaseModel):
    """Response model for statistical test results."""
    test_name: str
    statistic: float
    p_value: float
    critical_value: Optional[float]
    confidence_level: float
    interpretation: str

class StatisticalMetricsResponse(BaseModel):
    """Response model for statistical metrics."""
    descriptive_stats: Dict[str, float]
    correlation_matrix: Optional[Dict[str, Dict[str, float]]]
    outlier_scores: Optional[List[float]]

class AnalysisResponse(BaseModel):
    """Response model for analysis results."""
    analysis_id: str
    dataset_id: str
    user_id: str
    analysis_type: str
    status: str
    feature_columns: List[str]
    target_column: Optional[str]
    metrics: Optional[StatisticalMetricsResponse]
    statistical_tests: List[StatisticalTestResponse]
    insights: List[str]
    started_at: Optional[str]
    completed_at: Optional[str]
    execution_time_seconds: Optional[float]
    error_message: Optional[str]
    created_at: str
    
class AnalysisListResponse(BaseModel):
    """Response model for analysis list."""
    analyses: List[AnalysisResponse]
    total_count: int
    page: int
    page_size: int

class TaskResponse(BaseModel):
    """Response model for background tasks."""
    task_id: str
    status: str
    message: str
    analysis_id: str

# Router setup
router = APIRouter(
    prefix="/statistical-analysis",
    tags=["Statistical Analysis"],
    responses={
        400: {"description": "Bad Request"},
        404: {"description": "Not Found"},
        422: {"description": "Validation Error"},
        500: {"description": "Internal Server Error"},
    },
)

# Dependency injection setup (simplified for now)
def get_repository():
    """Get repository instance."""
    return InMemoryStatisticalAnalysisRepository()

def get_use_case(repository = Depends(get_repository)):
    """Get use case instance."""
    return ExecuteStatisticalAnalysisUseCase(repository)

def get_current_user():
    """Get current user (mock implementation)."""
    return UserId()

# Background task functions
async def execute_analysis_background(
    analysis_id: str,
    data_records: List[Dict[str, Any]],
    use_case: ExecuteStatisticalAnalysisUseCase
):
    """Execute analysis in background."""
    try:
        # Convert data to DataFrame
        data = pd.DataFrame(data_records)
        
        # Get analysis entity
        analysis = await use_case.get_analysis_by_id(
            StatisticalAnalysisId(value=UUID(analysis_id))
        )
        
        if analysis and analysis.status == "pending":
            # Execute analysis
            await use_case.repository.statistical_service.execute_analysis(analysis, data)
            
        logger.info("Background analysis completed", analysis_id=analysis_id)
        
    except Exception as e:
        logger.error("Background analysis failed", analysis_id=analysis_id, error=str(e))

# Endpoints
@router.post(
    "/",
    response_model=TaskResponse,
    summary="Create Statistical Analysis",
    description="""
    Create a new statistical analysis job.
    
    **Features:**
    - Descriptive statistics calculation
    - Hypothesis testing (t-test, chi-square, ANOVA, etc.)
    - Correlation analysis
    - Outlier detection
    - Automated insight generation
    
    **Analysis Types:**
    - descriptive_statistics: Basic statistical summaries
    - hypothesis_testing: Statistical hypothesis tests
    - correlation_analysis: Correlation matrices and significance tests
    - exploratory_analysis: Comprehensive EDA with visualizations
    """,
    responses={
        200: {"description": "Analysis created successfully"},
        400: {"description": "Invalid request parameters"},
        500: {"description": "Analysis creation failed"}
    }
)
async def create_analysis(
    request: CreateAnalysisRequest,
    data_request: AnalysisDataRequest,
    background_tasks: BackgroundTasks,
    use_case: ExecuteStatisticalAnalysisUseCase = Depends(get_use_case),
    current_user: UserId = Depends(get_current_user)
):
    """Create a new statistical analysis."""
    try:
        # Create analysis type
        analysis_type = AnalysisType(
            name=request.analysis_type,
            description=f"Statistical analysis of type: {request.analysis_type}",
            requires_target=request.target_column is not None
        )
        
        # Convert data to DataFrame for validation
        data = pd.DataFrame(data_request.data)
        
        # Validate feature columns exist
        missing_columns = set(request.feature_columns) - set(data.columns)
        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail=f"Missing columns in data: {list(missing_columns)}"
            )
        
        # Create analysis entity
        analysis = await use_case.execute(
            dataset_id=DatasetId(value=UUID(request.dataset_id)),
            user_id=current_user,
            analysis_type=analysis_type,
            data=data,
            feature_columns=request.feature_columns,
            analysis_params=request.analysis_params
        )
        
        # Add background task for processing
        background_tasks.add_task(
            execute_analysis_background,
            str(analysis.analysis_id.value),
            data_request.data,
            use_case
        )
        
        return TaskResponse(
            task_id=str(analysis.analysis_id.value),
            status="pending",
            message="Statistical analysis submitted for processing",
            analysis_id=str(analysis.analysis_id.value)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to create analysis", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Analysis creation failed: {str(e)}"
        )

@router.get(
    "/{analysis_id}",
    response_model=AnalysisResponse,
    summary="Get Statistical Analysis",
    description="Retrieve a specific statistical analysis by ID.",
    responses={
        200: {"description": "Analysis retrieved successfully"},
        404: {"description": "Analysis not found"}
    }
)
async def get_analysis(
    analysis_id: str,
    use_case: ExecuteStatisticalAnalysisUseCase = Depends(get_use_case)
):
    """Get analysis by ID."""
    try:
        analysis = await use_case.get_analysis_by_id(
            StatisticalAnalysisId(value=UUID(analysis_id))
        )
        
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        # Convert metrics to response model
        metrics_response = None
        if analysis.metrics:
            metrics_response = StatisticalMetricsResponse(
                descriptive_stats=analysis.metrics.descriptive_stats,
                correlation_matrix=analysis.metrics.correlation_matrix,
                outlier_scores=analysis.metrics.outlier_scores
            )
        
        # Convert tests to response models
        tests_response = [
            StatisticalTestResponse(
                test_name=test.test_name,
                statistic=test.statistic,
                p_value=test.p_value,
                critical_value=test.critical_value,
                confidence_level=test.confidence_level,
                interpretation=test.interpretation
            )
            for test in analysis.statistical_tests
        ]
        
        return AnalysisResponse(
            analysis_id=str(analysis.analysis_id.value),
            dataset_id=str(analysis.dataset_id.value),
            user_id=str(analysis.user_id.value),
            analysis_type=analysis.analysis_type.name,
            status=analysis.status,
            feature_columns=analysis.feature_columns,
            target_column=analysis.target_column,
            metrics=metrics_response,
            statistical_tests=tests_response,
            insights=analysis.insights,
            started_at=analysis.started_at.isoformat() if analysis.started_at else None,
            completed_at=analysis.completed_at.isoformat() if analysis.completed_at else None,
            execution_time_seconds=analysis.execution_time_seconds,
            error_message=analysis.error_message,
            created_at=analysis.created_at.isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get analysis", analysis_id=analysis_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve analysis: {str(e)}"
        )

@router.get(
    "/",
    response_model=AnalysisListResponse,
    summary="List Statistical Analyses",
    description="List statistical analyses with optional filtering.",
    responses={
        200: {"description": "Analyses retrieved successfully"}
    }
)
async def list_analyses(
    dataset_id: Optional[str] = Query(None, description="Filter by dataset ID"),
    status: Optional[str] = Query(None, description="Filter by status"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Items per page"),
    use_case: ExecuteStatisticalAnalysisUseCase = Depends(get_use_case),
    current_user: UserId = Depends(get_current_user)
):
    """List analyses with filtering and pagination."""
    try:
        # Get analyses based on filters
        if dataset_id:
            analyses = await use_case.get_analyses_by_dataset(
                DatasetId(value=UUID(dataset_id))
            )
        else:
            analyses = await use_case.get_analyses_by_user(current_user)
        
        # Filter by status if provided
        if status:
            analyses = [a for a in analyses if a.status == status]
        
        # Calculate pagination
        total_count = len(analyses)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_analyses = analyses[start_idx:end_idx]
        
        # Convert to response models
        analysis_responses = []
        for analysis in paginated_analyses:
            metrics_response = None
            if analysis.metrics:
                metrics_response = StatisticalMetricsResponse(
                    descriptive_stats=analysis.metrics.descriptive_stats,
                    correlation_matrix=analysis.metrics.correlation_matrix,
                    outlier_scores=analysis.metrics.outlier_scores
                )
            
            tests_response = [
                StatisticalTestResponse(
                    test_name=test.test_name,
                    statistic=test.statistic,
                    p_value=test.p_value,
                    critical_value=test.critical_value,
                    confidence_level=test.confidence_level,
                    interpretation=test.interpretation
                )
                for test in analysis.statistical_tests
            ]
            
            analysis_responses.append(AnalysisResponse(
                analysis_id=str(analysis.analysis_id.value),
                dataset_id=str(analysis.dataset_id.value),
                user_id=str(analysis.user_id.value),
                analysis_type=analysis.analysis_type.name,
                status=analysis.status,
                feature_columns=analysis.feature_columns,
                target_column=analysis.target_column,
                metrics=metrics_response,
                statistical_tests=tests_response,
                insights=analysis.insights,
                started_at=analysis.started_at.isoformat() if analysis.started_at else None,
                completed_at=analysis.completed_at.isoformat() if analysis.completed_at else None,
                execution_time_seconds=analysis.execution_time_seconds,
                error_message=analysis.error_message,
                created_at=analysis.created_at.isoformat()
            ))
        
        return AnalysisListResponse(
            analyses=analysis_responses,
            total_count=total_count,
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        logger.error("Failed to list analyses", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve analyses: {str(e)}"
        )

@router.delete(
    "/{analysis_id}",
    summary="Delete Statistical Analysis",
    description="Delete a statistical analysis by ID.",
    responses={
        200: {"description": "Analysis deleted successfully"},
        404: {"description": "Analysis not found"}
    }
)
async def delete_analysis(
    analysis_id: str,
    use_case: ExecuteStatisticalAnalysisUseCase = Depends(get_use_case)
):
    """Delete analysis by ID."""
    try:
        # Check if analysis exists
        analysis = await use_case.get_analysis_by_id(
            StatisticalAnalysisId(value=UUID(analysis_id))
        )
        
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        # Delete analysis
        await use_case.repository.delete(
            StatisticalAnalysisId(value=UUID(analysis_id))
        )
        
        return {"message": "Analysis deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete analysis", analysis_id=analysis_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete analysis: {str(e)}"
        )

@router.get(
    "/health",
    summary="Health Check",
    description="Check the health of the statistical analysis service.",
    responses={
        200: {"description": "Service is healthy"},
        503: {"description": "Service is unavailable"}
    }
)
async def health_check():
    """Health check endpoint."""
    try:
        # Simple health check
        return {
            "status": "healthy",
            "service": "statistical-analysis",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0"
        }
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(
            status_code=503,
            detail="Service unavailable"
        )